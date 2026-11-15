import csv  # noqa: EXE002
import glob
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from ahrs.filters import EKF, Madgwick
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.stats import pearsonr

# cache_dir = matplotlib.get_cachedir()
# font_cache_dir = os.path.join(cache_dir, "fontlist-v310.json")
# if os.path.exists(font_cache_dir):
#     os.remove(font_cache_dir)
#     print("Font cache removed.")
# else:
#     print("Font cache not found.")
figsize = (7, 6)
title = ""

# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = "Arial"
# plt.rcParams["font.size"] = 12
if sys.platform == "linux":
    split_char = "/"
else:
    split_char = "\\"


def rotation_matrix_from_vectors(vec_sensor, vec_world=[0, 0, 1]):
    """Compute the rotation matrix that aligns vec_sensor with vec_world."""
    vec_sensor = vec_sensor / np.linalg.norm(vec_sensor)
    vec_world = vec_world / np.linalg.norm(vec_world)

    v = np.cross(vec_sensor, vec_world)
    c = np.dot(vec_sensor, vec_world)
    s = np.linalg.norm(v)

    # Rotation matrix (Rodrigues' formula)
    if s < 1e-6:  # Parallel vectors
        return np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def low_pass_filter(data, cutoff_freq, fs, order=3):
    """
    Applies a low-pass filter to the given data.

    Args:
        data (array-like): Array of values corresponding to the timestamps.
        cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling frequency.
        order (int): Filter order (default: 3).

    Returns:
        numpy.ndarray: Filtered data.
    """
    # Calculate the Nyquist frequency
    nyq = 0.5 * fs
    # Normalize the cutoff frequency
    normal_cutoff = cutoff_freq / nyq
    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)
    # Apply the filter using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def csv_comment(comment):
    """
    Parse a comment line from the CSV file and return a dictionary of measures.
    The comment line is expected to start with a '#' character, followed by a space,
    """
    line = comment.replace("# ", "")
    comment = line.strip()
    measure = comment.split(";")[1:]
    measures = {
        m.split(":")[0]: n if (n := m.split(":")[1].strip()).isalpha() else float(n)
        for m in measure
    }
    return measures


def get_dataset(dataset_path: str) -> list[tuple[str, str]]:
    """
    Return a list of (mcd, arUco) file-path tuples found under the dataset folder.
    """

    dataset_path = str(dataset_path).rstrip("/\\")
    print(dataset_path)
    if len(dataset_path) == 0 or dataset_path == "":
        print("No folder selected")
        return []
    subjects = glob.glob("*", root_dir=dataset_path)
    subjects = list(map(int, subjects))
    subjects.sort()
    files = []
    for s in subjects:
        for i in range(4):
            mcd_file = f"{dataset_path}{split_char}{s}{split_char}mcd-{i}.csv"
            aruco_file = f"{dataset_path}{split_char}{s}{split_char}arUcos-{i}.csv"
            if not Path(mcd_file).exists() or not Path(aruco_file).exists():
                continue
            files.append((mcd_file, aruco_file))
    return files


def read_mcd_file(path: str, cutoff_freq=7, filter_order=6) -> dict[str, np.ndarray]:
    """
    Read the mcd file and return a dictionary with the data.

    Args:
        path (str): Path to the MCD CSV file.
        cutoff_freq (float): Cutoff frequency for the low-pass filter in Hz (default: 7).
        filter_order (int): Order of the Butterworth filter (default: 3).
    """
    # open the file and split it into dictionaries
    with open(path, "r") as file:
        measures = csv_comment(file.readline())
        mcd_file = [row for row in csv.DictReader(file, delimiter=";")]
    # mcd_file = mcd_file[100:]
    # convert the data into a dictionary of float numpy arrays
    mcd_data = {
        key: np.array(
            [np.array(data[key].split(","), dtype=float) for data in mcd_file]
        )
        for key in mcd_file[0].keys()
    }
    # 'flatten' the time array so that it is one-dimensional
    mcd_data["time"] = mcd_data["time"].flatten()
    # add the measurements dictionary
    mcd_data["measures"] = measures
    # get the sampling frequency
    fs = 1000 / np.mean(np.diff(mcd_data["time"]))
    # fs=100
    # convert the data to units (rad/s and m/s^2) and apply a low-pass filter
    for i in ["RF", "RT", "LF", "LT"]:

        mcd_data[f"gyro_{i}"] = (mcd_data[f"gyro_{i}"] / 16.4) * (np.pi / 180)
        mcd_data[f"acceleration_{i}"] = (mcd_data[f"acceleration_{i}"] / 16_384) * 9.81
        if cutoff_freq is not None:
            mcd_data[f"gyro_{i}"] = np.array(
                [
                    low_pass_filter(
                        mcd_data[f"gyro_{i}"].T[n],
                        cutoff_freq,
                        fs,
                        order=filter_order,
                    )
                    for n in range(3)
                ]
            ).T

            mcd_data[f"acceleration_{i}"] = np.array(
                [
                    low_pass_filter(
                        mcd_data[f"acceleration_{i}"].T[n],
                        cutoff_freq,
                        fs,
                        order=filter_order,
                    )
                    for n in range(3)
                ]
            ).T

    return mcd_data


def read_aruco_file(path: str) -> dict[str, np.ndarray]:
    """
    Read the arUco file and return a dictionary with the data.
    """
    # open the file and split it into dictionaries
    with open(path, "r") as file:
        aruco_file = [row for row in csv.DictReader(file, delimiter=";")]
    # extract the keys
    aruco_keys = aruco_file[0].keys()
    # group the data of each key into a list
    aruco_data_no_processed = {
        key: [data[key] for data in aruco_file] for key in aruco_keys
    }
    # extract all the timestamps into an integer array
    aruco_time = np.array([row["time"] for row in aruco_file], dtype=int)
    # bind each datum to its timestamp
    aruco_data = {
        key: {
            time: np.array(data.split(","), dtype=float)
            for time, data in zip(aruco_time, aruco_data_no_processed[key])
            if data != "" and time >= 0
        }
        for key in aruco_keys
        if key != "time"
    }
    # split timestamps and data into two arrays, still linked by index
    for key, values in aruco_data.items():
        aruco_data[key] = {
            "time": np.array(list(values.keys()), dtype=int),
            "data": np.array([values[time] for time in values.keys()]),
        }

    return aruco_data


def adjust_aruco_data(
    aruco_data: dict[str, np.ndarray], direction
) -> dict[str, np.ndarray]:
    x_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 0])
    )
    y_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 1])
    )
    z_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 2])
    )
    init = np.array([x_init, y_init, z_init])

    aruco_data["hip_position"]["data"] -= init
    aruco_data["R_knee_position"]["data"] -= init
    aruco_data["L_knee_position"]["data"] -= init
    aruco_data["R_ankle_position"]["data"] -= init
    aruco_data["L_ankle_position"]["data"] -= init

    return aruco_data


def transformation_matrix(position, rotation) -> np.ndarray:
    """
    Build a 4x4 transformation matrix from a 3D position and a quaternion.

    Args:
        position (list or numpy.ndarray): 3-element list or numpy array [x, y, z] with the position.
        rotation: --
    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """

    # 1. Rotation matrix
    rotation_matrix = rotation.as_matrix()

    # 2. Translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # 3. Full transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix


def apply_transformation(transformation_matrix, point):
    """
    Apply a transformation matrix to a point.

    Args:
        transformation_matrix (numpy.ndarray): 4x4 transformation matrix.
        point (list or numpy.ndarray): 3-element list or numpy array [x, y, z] with the point.

    Returns:
        numpy.ndarray: 3-element numpy array [x', y', z'] with the new position of the point.
    """

    # 1. Point in homogeneous coordinates
    homogeneous_point = np.append(point, 1)  # add 1 to the end of the array
    # change the array to be a 4x1 array.
    homogeneous_point = homogeneous_point.reshape((4, 1))

    # 2. Matrix multiplication
    transformed_point = np.dot(transformation_matrix, homogeneous_point)

    # 3. Extract the new position
    # take the first 3 elements, and flatten the array.
    new_point = transformed_point[:3].flatten()

    return new_point


def get_pose(quats, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pose = {}
    # transformation matrix of the hip
    T_h = transformation_matrix([0, 0, 0], quats["hip"])
    # hip points
    pose["hip_r"] = apply_transformation(T_h, points["hip_r"])
    pose["hip_l"] = apply_transformation(T_h, points["hip_l"])
    # Transformation matrix of the knees (relative to the hip)
    # made of the translation (position) of the hip and the rotation of the femur
    T_kr = transformation_matrix(pose["hip_r"], quats["rf"])
    T_kl = transformation_matrix(pose["hip_l"], quats["lf"])
    # knee points
    pose["knee_r"] = apply_transformation(T_kr, points["knee_r"])
    pose["knee_l"] = apply_transformation(T_kl, points["knee_l"])
    # Transformation matrix of the ankles (relative to the knees)
    # made of the translation (position) of the knee and the rotation of the tibia
    T_ar = transformation_matrix(pose["knee_r"], quats["rt"])
    T_al = transformation_matrix(pose["knee_l"], quats["lt"])
    pose["ankle_r"] = apply_transformation(T_ar, points["ankle_r"])
    pose["ankle_l"] = apply_transformation(T_al, points["ankle_l"])

    return pose


def get_all_poses(quats, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(quats["rf"])
    pose = {
        "hip_r": np.empty((n, 3)),
        "hip_l": np.empty((n, 3)),
        "knee_r": np.empty((n, 3)),
        "knee_l": np.empty((n, 3)),
        "ankle_r": np.empty((n, 3)),
        "ankle_l": np.empty((n, 3)),
    }
    for i in range(n):
        current_quats = {
            "rf": quats["rf"][i],
            "lf": quats["lf"][i],
            "rt": quats["rt"][i],
            "lt": quats["lt"][i],
            "hip": quats["hip"][i],
        }
        current_pose = get_pose(current_quats, points)
        pose["hip_r"][i] = current_pose["hip_r"]
        pose["hip_l"][i] = current_pose["hip_l"]
        pose["knee_r"][i] = current_pose["knee_r"]
        pose["knee_l"][i] = current_pose["knee_l"]
        pose["ankle_r"][i] = current_pose["ankle_r"]
        pose["ankle_l"][i] = current_pose["ankle_l"]
    return pose


def reorient_data(
    accel: np.ndarray, gyro: np.ndarray, calibration_samples=100
) -> tuple[np.ndarray, np.ndarray]:
    rest = np.mean(accel[:calibration_samples], axis=0)
    rest /= np.linalg.norm(rest)
    accel_R = rotation_matrix_from_vectors(rest)
    reoriented_accel = (accel_R @ accel.T).T
    reoriented_gyro = (accel_R @ gyro.T).T

    return reoriented_accel, reoriented_gyro


def get_orientation(
    mcd_data,
    pca_percentile=5,
    madgwick_gain=0.01,
    slerp_clip_min=0.3,
    slerp_clip_max=0.7,
    calibration_samples=60,
    fusion_filter="madgwick",
) -> dict[str, np.ndarray]:
    """
    Get the orientation of the sensors using the Madgwick filter.
    """
    reoriented_data = {
        "rf": reorient_data(
            mcd_data["acceleration_RF"],
            mcd_data["gyro_RF"],
            calibration_samples=calibration_samples,
        ),
        "lf": reorient_data(
            mcd_data["acceleration_LF"],
            mcd_data["gyro_LF"],
            calibration_samples=calibration_samples,
        ),
        "rt": reorient_data(
            mcd_data["acceleration_RT"],
            mcd_data["gyro_RT"],
            calibration_samples=calibration_samples,
        ),
        "lt": reorient_data(
            mcd_data["acceleration_LT"],
            mcd_data["gyro_LT"],
            calibration_samples=calibration_samples,
        ),
    }
    # # print(mcd_data.keys())
    # reoriented_data = {
    #     "rf": (mcd_data["acceleration_RF"], mcd_data["gyro_RF"]),
    #     "lf": (mcd_data["acceleration_LF"], mcd_data["gyro_LF"]),
    #     "rt": (mcd_data["acceleration_RT"], mcd_data["gyro_RT"]),
    #     "lt": (mcd_data["acceleration_LT"], mcd_data["gyro_LT"]),
    # }
    reoriented_accel = {
        "rf": reoriented_data["rf"][0],
        "lf": reoriented_data["lf"][0],
        "rt": reoriented_data["rt"][0],
        "lt": reoriented_data["lt"][0],
    }
    reoriented_gyro = {
        "rf": reoriented_data["rf"][1],
        "lf": reoriented_data["lf"][1],
        "rt": reoriented_data["rt"][1],
        "lt": reoriented_data["lt"][1],
    }
    # creation of the complementary filters
    s_p = 1000 / np.mean(np.diff(mcd_data["time"]))
    # print(s_p)
    # s_p = 100

    if fusion_filter == "madgwick":
        quats = {
            "rf": R.from_quat(
                Madgwick(
                    reoriented_gyro["rf"],
                    reoriented_accel["rf"],
                    frequency=s_p,
                    gain=madgwick_gain,
                ).Q,
                scalar_first=True,
            ),
            "lf": R.from_quat(
                Madgwick(
                    reoriented_gyro["lf"],
                    reoriented_accel["lf"],
                    frequency=s_p,
                    gain=madgwick_gain,
                ).Q,
                scalar_first=True,
            ),
            "rt": R.from_quat(
                Madgwick(
                    reoriented_gyro["rt"],
                    reoriented_accel["rt"],
                    frequency=s_p,
                    gain=madgwick_gain,
                ).Q,
                scalar_first=True,
            ),
            "lt": R.from_quat(
                Madgwick(
                    reoriented_gyro["lt"],
                    reoriented_accel["lt"],
                    frequency=s_p,
                    gain=madgwick_gain,
                ).Q,
                scalar_first=True,
            ),
            "hip": np.empty(0),
        }
    else:
        quats = {
            "rf": R.from_quat(
                EKF(
                    gyr=reoriented_gyro["rf"],
                    acc=reoriented_accel["rf"],
                    frequency=s_p,
                    frame="NED",
                ).Q,
                scalar_first=True,
            ),
            "lf": R.from_quat(
                EKF(
                    gyr=reoriented_gyro["lf"],
                    acc=reoriented_accel["lf"],
                    frequency=s_p,
                    frame="NED",
                ).Q,
                scalar_first=True,
            ),
            "rt": R.from_quat(
                EKF(
                    gyr=reoriented_gyro["rt"],
                    acc=reoriented_accel["rt"],
                    frequency=s_p,
                    frame="NED",
                ).Q,
                scalar_first=True,
            ),
            "lt": R.from_quat(
                EKF(
                    gyr=reoriented_gyro["lt"],
                    acc=reoriented_accel["lt"],
                    frequency=s_p,
                    frame="NED",
                ).Q,
                scalar_first=True,
            ),
            "hip": np.empty(0),
        }
    for key in quats.keys():
        if key == "hip":
            continue

        # ADVANCED METHOD: PCA on the gyroscope to find the knee hinge axis
        gyro_data = reoriented_gyro[key]

        # Filter to use only the moments of largest motion (swing phase of gait)
        magnitudes = np.linalg.norm(gyro_data, axis=1)
        threshold = np.percentile(
            magnitudes, pca_percentile
        )  # Use only the top (100-pca_percentile)% most active samples
        movimiento = gyro_data[magnitudes > threshold]

        if len(movimiento) > 10:
            # Principal Component Analysis (PCA) via the NumPy covariance matrix
            cov_matrix = np.cov(movimiento.T)
            # Enforce strict symmetry to avoid numerical errors that lead to complex eigenvalues
            cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

            if np.isfinite(cov_matrix).all():
                # For real symmetric matrices, eigh guarantees real eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # The principal axis is the largest eigenvalue (functional flexion/extension axis)
                principal_axis = np.real(eigenvectors[:, np.argmax(eigenvalues)])

                # Project onto the horizontal plane (XY), ignoring gravity
                axis_xy = np.real(principal_axis[:2]).astype(float)
                norm_xy = np.linalg.norm(axis_xy)

                if np.isfinite(norm_xy) and norm_xy > 1e-6:
                    axis_xy = axis_xy / norm_xy

                    # Force the vector to point towards positive Y to avoid 180° flips from the PCA sign ambiguity
                    if axis_xy[1] < 0:
                        axis_xy = -axis_xy

                    yaw_angle = float(np.arctan2(axis_xy[1], axis_xy[0]))

                    # Align this principal axis (sagittal hinge) with the global Y axis (pi/2)
                    correction_yaw = float((np.pi / 2) - yaw_angle)
                    if not np.isfinite(correction_yaw):
                        correction_yaw = 0.0
                else:
                    correction_yaw = 0.0
            else:
                correction_yaw = 0.0
        else:
            correction_yaw = 0.0

        rot_yaw_corr = R.from_euler("Z", correction_yaw)

        quats[key] = rot_yaw_corr * quats[key]
    # angles = quats["lf"].as_euler("zyx", degrees=True)
    # angles = quats["lf"].as_rotvec(degrees=True)
    # plt.plot(angles[:, 0], "r")
    # plt.plot(angles[:, 1], "g")
    # plt.plot(angles[:, 2], "b")
    # plt.show()

    # Apply a dynamic kinematic model to the hip (weighted Slerp)
    hip_quats_list = []
    for i in range(len(mcd_data["time"])):
        # Determine which leg is in stance (lower angular velocity)
        vel_rf = np.linalg.norm(reoriented_gyro["rf"][i])
        vel_lf = np.linalg.norm(reoriented_gyro["lf"][i])
        total_vel = vel_rf + vel_lf

        # If the right leg (rf) is static, its velocity is low.
        # Slerp interpolates: 0.0 -> rf, 1.0 -> lf
        if total_vel < 1e-5:
            weight = 0.5
        else:
            # Velocity-based weighting: if rf is still, the weight tends to 0.
            weight = vel_rf / total_vel

        # Clip the weight (0.3 to 0.7) so the hip is never fused 100% with a single femur, avoiding rigidity
        weight = np.clip(weight, slerp_clip_min, slerp_clip_max)

        slerp = Slerp([0, 1], R.concatenate([quats["rf"][i], quats["lf"][i]]))
        hip_quats_list.append(slerp(weight).as_quat())

    quats["hip"] = R.from_quat(np.array(hip_quats_list))

    # rot_rk = np.empty((0, 3))
    # rot_lk = np.empty((0, 3))
    rot_r = (quats["hip"].inv() * quats["rf"]).as_euler("xyz", degrees=True)
    rot_l = (quats["hip"].inv() * quats["lf"]).as_euler("zxy", degrees=True)
    # print(f"rot q rf: {rot_r[100:-100].mean(axis=0)}")
    # for a in rot_r:
    #     print(a)
    # plt.plot(rot_r)
    # plt.show()
    R.from_euler("z", -np.mean(rot_r[:, 0]), degrees=True).as_quat(scalar_first=True)
    R.from_euler("z", -np.mean(rot_l[:, 0]), degrees=True).as_quat(scalar_first=True)
    # for i in range(len(quats['rf'])):
    #     quats['rf'][i] = multiply(fix_rot_r, quats['rf'][i])
    #     quats['rt'][i] = multiply(fix_rot_r, quats['rt'][i])
    #     # quats['rt'][i] = multiply(fix_rot_rk, quats['rt'][i])
    #     quats['lf'][i] = multiply(fix_rot_l, quats['lf'][i])
    #     quats['lt'][i] = multiply(fix_rot_l, quats['lt'][i])
    # quats['lt'][i] = multiply(fix_rot_lk, quats['lt'][i])

    # p_rf = np.array([R.from_quat(r, scalar_first=True).as_matrix()
    #                  @ [0, 0, -1] for r in quats['rf']])
    # ax = plt.subplot()
    # ax.plot(p_rf[200:-200, 0], p_rf[200:-200, 1], '.')
    # ax.set_xlim(-.6, .6)
    # ax.set_ylim(-.6, .6)
    # plt.plot(quats['rf'].as_euler('ZXY'))

    # ph = np.array([r.apply([1, 0, 0]) for r in quats["hip"]])
    # mh = ph[:, 1] / ph[:, 0]
    # for b in ["rf", "lf", "rt", "lt"]:
    #     p = np.array([r.apply([0, 0, -1]) for r in quats[b]])
    #     x = p[200:-200, 0].reshape(-1, 1)
    #     y = p[200:-200, 1]
    #     model = LinearRegression()
    #     model.fit(x, y)
    #     m1 = model.coef_[0]
    #     angle = np.arctan(m1) + np.arctan(mh)
    #     # angle = np.arctan(m1)
    #     # print('slope: ', model.coef_[0])
    #     # print('angle: ', np.arctan(model.coef_[0]))
    #     # rot = R.from_euler('z', [-np.arctan(model.coef_[0])])
    #     rot = R.from_euler("z", -angle)
    # quats[b] = rot * quats[b]
    # ph = np.array([r.apply([1, 0, 0]) for r in quats["hip"]])
    # mh = ph[:, 1] / ph[:, 0]
    # for b in ["rf", "lf", "rt", "lt"]:
    #     p = np.array([r.apply([0, 0, -1]) for r in quats[b]])
    #     x = p[:, 0].reshape(-1, 1)
    #     y = p[:, 1]
    #     model = LinearRegression()
    #     model.fit(x[200:300], y[200:300])
    #     init_correction_angle = model.coef_[0]
    #     for i in range(1, len(quats[b]), 1):
    #         # m1 = model.coef_[0]
    #         angle = np.arctan(init_correction_angle) + np.arctan(mh[i])
    #         # angle = np.arctan(m1)
    #         # print('slope: ', model.coef_[0])
    #         # print('angle: ', np.arctan(model.coef_[0]))
    #         # rot = R.from_euler('z', [-np.arctan(model.coef_[0])])
    #         rot = R.from_euler("z", -angle)
    #         quats[b] = rot * quats[b]
    # plt.plot(quats['rf'].as_euler('ZXY'), '--')
    # plt.show()
    return quats


def mcd_reconstruction(
    mcd_data: dict[str, np.ndarray],
    pca_percentile=5,
    madgwick_gain=0.01,
    slerp_clip_min=0.3,
    slerp_clip_max=0.7,
    calibration_samples=60,
    fusion_filter="madgwick",
    pivot_height_threshold=0.01,
    swing_velocity_threshold=0.5,
) -> dict[str, np.ndarray]:
    direction = mcd_data["measures"]["direction"]

    points = {
        "hip_r": np.array([0, mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_r": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_r": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
        "hip_l": np.array([0, -mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_l": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_l": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
    }
    quats = get_orientation(
        mcd_data,
        pca_percentile=pca_percentile,
        madgwick_gain=madgwick_gain,
        slerp_clip_min=slerp_clip_min,
        slerp_clip_max=slerp_clip_max,
        calibration_samples=calibration_samples,
        fusion_filter=fusion_filter,
    )

    pose = get_all_poses(quats, points)
    # plt.show()
    n = len(mcd_data["time"])
    reconstruction = {
        "hip_r": np.empty((n, 3)),
        "hip_l": np.empty((n, 3)),
        "knee_r": np.empty((n, 3)),
        "knee_l": np.empty((n, 3)),
        "ankle_r": np.empty((n, 3)),
        "ankle_l": np.empty((n, 3)),
    }

    last_T_al = np.eye(4)
    last_T_ar = np.eye(4)
    # plt.plot(mcd_data['time'][1:], np.diff(
    #     pose['ankle_l'][:, 0]), label='ankle_l')
    # plt.plot(mcd_data['time'][1:], np.diff(
    #     pose['ankle_r'][:, 0]), label='ankle_r')
    # plt.show()
    np.diff(pose["ankle_r"][:, 0])
    np.diff(pose["ankle_l"][:, 0])
    for i in range(n):
        if i == 0:
            z_compare = 1 if direction == "R" else 0
        else:
            # 1. Kinematics (Z height difference)
            diff_z = pose["ankle_l"][i][2] - pose["ankle_r"][i][2]

            # 2. Dynamics (gyroscope magnitude of the shanks)
            # During stance, the pivot leg barely rotates
            mag_gyro_rt = np.linalg.norm(mcd_data["gyro_RT"][i])
            mag_gyro_lt = np.linalg.norm(mcd_data["gyro_LT"][i])

            # If the height difference is obvious (> 1.5 cm), trust the position
            if abs(diff_z) > pivot_height_threshold:
                z_compare = diff_z
            else:
                # Transition or double-support phase: the leg that moves less is the pivot.
                # z_compare > 0 => right pivot, so we compute left - right.
                z_compare = mag_gyro_lt - mag_gyro_rt

        v_rt = mcd_data["gyro_RT"][i, 1]
        v_lt = mcd_data["gyro_LT"][i, 1]

        # Extra filter: if the presumed pivot is swinging strongly forward,
        # force the switch to the opposite foot
        if z_compare <= 0:  # Presumed left pivot
            if v_lt > swing_velocity_threshold and v_lt > v_rt:
                z_compare = 1
        else:  # Presumed right pivot
            if v_rt > swing_velocity_threshold and v_rt > v_lt:
                z_compare = -1

        # translate the ankle to the origin
        if z_compare > 0:
            T_a_2_o = transformation_matrix(
                pose["ankle_r"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_ar
        else:
            T_a_2_o = transformation_matrix(
                pose["ankle_l"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_al

        T_acu = np.linalg.inv(T_a_2_o) @ T_pivot

        reconstruction["hip_r"][i] = apply_transformation(T_acu, pose["hip_r"][i])
        reconstruction["hip_l"][i] = apply_transformation(T_acu, pose["hip_l"][i])
        reconstruction["knee_r"][i] = apply_transformation(T_acu, pose["knee_r"][i])
        reconstruction["knee_l"][i] = apply_transformation(T_acu, pose["knee_l"][i])
        reconstruction["ankle_r"][i] = apply_transformation(T_acu, pose["ankle_r"][i])
        reconstruction["ankle_l"][i] = apply_transformation(T_acu, pose["ankle_l"][i])

        last_T_al = transformation_matrix(
            reconstruction["ankle_l"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
        )
        last_T_ar = transformation_matrix(
            reconstruction["ankle_r"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
        )

    # fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 4))
    # fig.suptitle(title)
    # # ax.plot(rot_rk[:, 0], label='Flexion-Extension Z')
    # # ax.plot(rot_rk[:, 1], label='Flexion-Extension X')
    # # ax.plot(rot_rk[:, 2], label='Flexion-Extension Y')
    # ax[1, 1].set_title('Right Hip')
    # ax[1, 1].plot(rot_r[:,0], label='rf Z')
    # ax[1, 1].plot(rot_r[:,1], label='rf x')
    # ax[1, 1].plot(rot_r[:,2], label='rf y')
    # ax[1, 1].legend()
    # ax[1, 1].grid(1)
    # ax[2, 1].set_title('Left Hip')
    # ax[2, 1].grid(1)
    # ax[2, 1].plot(rot_l[:,0], label='lf Z')
    # ax[2, 1].plot(rot_l[:,1], label='lf x')
    # ax[2, 1].plot(rot_l[:,2], label='lf y')
    # ax[2, 1].legend()
    # ax[1, 0].set_title('Right Knee')
    # ax[1, 0].grid(1)
    # ax[1, 0].plot(rot_rk[:,0], label='rk Z')
    # ax[1, 0].plot(rot_rk[:,1], label='rk x')
    # ax[1, 0].plot(rot_rk[:,2], label='rk y')
    # ax[1, 0].legend()
    # ax[2, 0].set_title('Left Knee')
    # ax[2, 0].grid(1)
    # ax[2, 0].plot(rot_lk[:,0], label='lk Z')
    # ax[2, 0].plot(rot_lk[:,1], label='lk x')
    # ax[2, 0].plot(rot_lk[:,2], label='lk y')
    # ax[2, 0].legend()

    return reconstruction


def euclidean_error_3Dpoint(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean error between two 3D points.

    Args:
        a (numpy.ndarray): First 3D point.
        b (numpy.ndarray): Second 3D point.

    Returns:
        float: Euclidean error.
    """
    return np.linalg.norm(a - b)


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Mean absolute error.
    """
    euclidean = np.linalg.norm(a - b, axis=1)
    return np.abs(euclidean).mean()


def interpolate_point(t, point):
    return [
        interp1d(t, point[:, 0]),
        interp1d(t, point[:, 1]),
        interp1d(t, point[:, 2]),
    ]


def homogenize_3dpoint_data(data_a, data_b, timestamp_a, timestamp_b):
    a_int = interpolate_point(timestamp_a, data_a)
    b_int = interpolate_point(timestamp_b, data_b)
    t_min = max(min(timestamp_a), min(timestamp_b))
    t_max = min(max(timestamp_a), max(timestamp_b))
    c1 = timestamp_a >= t_min
    c2 = timestamp_a <= t_max
    t = timestamp_a[c1 & c2]

    a_out = np.array([a_int[i](t) for i in range(3)]).T
    b_out = np.array([b_int[i](t) for i in range(3)]).T
    return t, a_out, b_out


def measures_calculation(aruco, direction):
    femur_common_time = set(aruco[f"knee_{direction.lower()}"]["time"]).intersection(
        set(aruco[f"hip_{direction.lower()}"]["time"])
    )
    femur_lengths = []
    tibia_lengths = []
    for t in femur_common_time:
        knee_idx = np.where(aruco[f"knee_{direction.lower()}"]["time"] == t)
        knee = aruco[f"knee_{direction.lower()}"]["data"][knee_idx][0]
        hip_idx = np.where(aruco[f"hip_{direction.lower()}"]["time"] == t)
        hip = aruco[f"hip_{direction.lower()}"]["data"][hip_idx][0]

        f_length = np.sqrt(
            (knee[0] - hip[0]) ** 2 + (knee[1] - hip[1]) ** 2 + (knee[2] - hip[2]) ** 2
        )

        femur_lengths.append(f_length)

    tibia_common_time = set(aruco[f"knee_{direction.lower()}"]["time"]).intersection(
        set(aruco[f"ankle_{direction.lower()}"]["time"])
    )
    for t in tibia_common_time:
        knee_idx = np.where(aruco[f"knee_{direction.lower()}"]["time"] == t)
        knee = aruco[f"knee_{direction.lower()}"]["data"][knee_idx][0]
        ankle_idx = np.where(aruco[f"ankle_{direction.lower()}"]["time"] == t)
        ankle = aruco[f"ankle_{direction.lower()}"]["data"][ankle_idx][0]
        t_length = np.sqrt(
            (knee[0] - ankle[0]) ** 2
            + (knee[1] - ankle[1]) ** 2
            + (knee[2] - ankle[2]) ** 2
        )
        tibia_lengths.append(t_length)

    return np.mean(femur_lengths), np.mean(tibia_lengths)


def compute_displacement_velocity(positions, timestamps_ms, trim_ms=1000):
    """
    Compute the walking speed (m/s) from 3D positions and timestamps.
    Trims the still periods at the start and end of the capture.

    Args:
        positions (np.ndarray): (N, 3) array of 3D positions.
        timestamps_ms (np.ndarray): (N,) array of timestamps in milliseconds.
        trim_ms (float): Milliseconds to trim at the start and end to avoid
                         static periods that would bias the average.

    Returns:
        dict: Dictionary with:
            - 'velocities': array of instantaneous speeds (m/s)
            - 'timestamps': array of the corresponding timestamps (ms)
            - 'mean_velocity': mean speed (m/s)
            - 'std_velocity': standard deviation of the speed
            - 'var_velocity': variance of the speed
    """
    # Trim still periods at the start and end
    t_min = timestamps_ms[0] + trim_ms
    t_max = timestamps_ms[-1] - trim_ms
    mask = (timestamps_ms >= t_min) & (timestamps_ms <= t_max)

    trimmed_pos = positions[mask]
    trimmed_t = timestamps_ms[mask]

    if len(trimmed_t) < 2:
        return {
            "velocities": np.array([]),
            "timestamps": np.array([]),
            "mean_velocity": 0.0,
            "std_velocity": 0.0,
            "var_velocity": 0.0,
        }

    # Position and time differences
    dp = np.diff(trimmed_pos, axis=0)  # (N-1, 3)
    dt = np.diff(trimmed_t) / 1000.0  # convert ms to seconds, (N-1,)

    # Instantaneous speed (displacement magnitude / dt)
    speeds = np.linalg.norm(dp, axis=1) / dt  # m/s

    # Discard unrealistic speeds (> 5 m/s for human gait)
    valid = speeds < 5.0
    speeds = speeds[valid]
    t_out = trimmed_t[1:][valid]

    return {
        "velocities": speeds,
        "timestamps": t_out,
        "mean_velocity": np.mean(speeds) if len(speeds) > 0 else 0.0,
        "std_velocity": np.std(speeds) if len(speeds) > 0 else 0.0,
        "var_velocity": np.var(speeds) if len(speeds) > 0 else 0.0,
    }


def compute_knee_angle(hip_pos, knee_pos, ankle_pos):
    """
    Compute the knee flexion angle from the hip, knee and ankle
    positions.

    Returns:
        float: Knee flexion angle in degrees (0° = extended).
    """
    vec_femur = hip_pos - knee_pos
    vec_tibia = ankle_pos - knee_pos
    norm_f = np.linalg.norm(vec_femur)
    norm_t = np.linalg.norm(vec_tibia)
    if norm_f < 1e-10 or norm_t < 1e-10:
        return 0.0
    cos_angle = np.dot(vec_femur, vec_tibia) / (norm_f * norm_t)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return 180.0 - np.degrees(np.arccos(cos_angle))


def compute_knee_angles_series(hip_pos, knee_pos, ankle_pos):
    """Compute the time series of knee angles in degrees."""
    n = min(len(hip_pos), len(knee_pos), len(ankle_pos))
    return np.array(
        [compute_knee_angle(hip_pos[i], knee_pos[i], ankle_pos[i]) for i in range(n)]
    )


def detect_stride_lengths(
    ankle_positions, timestamps_ms, min_distance_samples=50, prominence=0.005
):
    """
    Detect stride lengths from heel strikes (minima of the ankle Z).

    Args:
        ankle_positions (np.ndarray): Ankle positions (N, 3).
        timestamps_ms (np.ndarray): Timestamps in ms.
        min_distance_samples (int): Minimum distance between peaks in samples.
        prominence (float): Minimum prominence to detect a heel strike.

    Returns:
        tuple: (stride_lengths_m, stride_times_ms)
    """
    z = ankle_positions[:, 2]
    # Heel strikes = minima in Z → invert to use find_peaks
    neg_z = -z
    peaks, _ = find_peaks(neg_z, distance=min_distance_samples, prominence=prominence)

    if len(peaks) < 2:
        return np.array([]), np.array([])

    stride_lengths = []
    stride_times = []
    # Each consecutive pair of heel strikes of the same foot = 1 stride
    # Use alternate pairs (i, i+2) for strides of the same foot
    for i in range(len(peaks) - 2):
        p1 = ankle_positions[peaks[i]]
        p2 = ankle_positions[peaks[i + 2]]
        # Horizontal (X-Y) distance between heel strikes of the same foot
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        stride_len = np.sqrt(dx**2 + dy**2)
        if stride_len > 0.1 and stride_len < 3.0:  # Discard unrealistic values
            stride_lengths.append(stride_len)
            stride_times.append(timestamps_ms[peaks[i]])

    return np.array(stride_lengths), np.array(stride_times)


def process_files(
    files: tuple[str, str],
    use_3d=False,
    cutoff_freq=7,
    filter_order=6,
    pca_percentile=5,
    madgwick_gain=0.01,
    slerp_clip_min=0.3,
    slerp_clip_max=0.7,
    calibration_samples=60,
    fusion_filter="madgwick",
    pivot_height_threshold=0.01,
    swing_velocity_threshold=0.5,
) -> any:
    # Read the MCD (inertial sensors) and ArUco (ground truth) files.
    mcd_data = read_mcd_file(
        files[0], cutoff_freq=cutoff_freq, filter_order=filter_order
    )
    aruco_data = read_aruco_file(files[1])

    # Determine the walking direction and the hidden (occluded) leg.
    direction = mcd_data["measures"]["direction"]
    hidden_leg = "L" if direction == "R" else "R"

    # _, res = check_sudden_variations(aruco_data, direction, 1000, 0.15)
    # all_jumps = []
    # for res in res.values():
    #     all_jumps.extend(res.get("jumps_list", []))

    # total_jumps = len(all_jumps)

    # if total_jumps > 5:
    #     print(files[1])
    #     return None

    # Shift the ArUco data so that it starts at the origin (0,0,0).
    aruco_data = adjust_aruco_data(aruco_data, direction)
    # Rearrange the ArUco data into a more convenient format for processing.
    aruco_processed = {
        "hip_r": aruco_data["hip_position"],
        "hip_l": aruco_data["hip_position"],
        "knee_r": aruco_data["R_knee_position"],
        "knee_l": aruco_data["L_knee_position"],
        "ankle_r": aruco_data["R_ankle_position"],
        "ankle_l": aruco_data["L_ankle_position"],
    }

    for key in aruco_processed.keys():
        aruco_processed[key]["data"][:, 1] = 0

    # Compute the femur and tibia lengths from the ArUco data.
    femur_length, tibia_length = measures_calculation(aruco_processed, direction)
    # Update the measurements in the MCD data with the computed lengths.
    mcd_data["measures"]["femur_length"] = femur_length
    mcd_data["measures"]["tibia_length"] = tibia_length
    if (
        "hip_width" not in mcd_data["measures"]
        or mcd_data["measures"]["hip_width"] <= 0.05
    ):
        mcd_data["measures"]["hip_width"] = 0.25

    # Reconstruct the skeleton from the inertial-sensor data.
    reconstruction = mcd_reconstruction(
        mcd_data,
        pca_percentile=pca_percentile,
        madgwick_gain=madgwick_gain,
        slerp_clip_min=slerp_clip_min,
        slerp_clip_max=slerp_clip_max,
        calibration_samples=calibration_samples,
        fusion_filter=fusion_filter,
        pivot_height_threshold=pivot_height_threshold,
        swing_velocity_threshold=swing_velocity_threshold,
    )
    # reconstruction = mcd_reconstruction_are(mcd_data)
    # Define a target rotation angle to align the reconstruction with the global coordinate frame.
    z_target = 0 if direction == "L" else np.pi
    # Build a transformation matrix to apply the rotation correction.
    T_fix = transformation_matrix([0, 0, 0], R.from_euler("ZXY", [z_target, 0, 0]))
    # Apply the transformation matrix to every point of the reconstruction.
    for i in range(reconstruction["hip_r"].shape[0]):
        reconstruction["hip_r"][i] = apply_transformation(
            T_fix, reconstruction["hip_r"][i]
        )
        reconstruction["hip_l"][i] = apply_transformation(
            T_fix, reconstruction["hip_l"][i]
        )
        reconstruction["knee_r"][i] = apply_transformation(
            T_fix, reconstruction["knee_r"][i]
        )
        reconstruction["knee_l"][i] = apply_transformation(
            T_fix, reconstruction["knee_l"][i]
        )
        reconstruction["ankle_r"][i] = apply_transformation(
            T_fix, reconstruction["ankle_r"][i]
        )
        reconstruction["ankle_l"][i] = apply_transformation(
            T_fix, reconstruction["ankle_l"][i]
        )

    # Homogenize the data so that the MCD and ArUco data share the same timestamps.
    homogeneous_data = {}
    for key in reconstruction.keys():
        # if key in [f'hip_{hidden_leg.lower()}', f'knee_{hidden_leg.lower()}', f'ankle_{hidden_leg.lower()}']:
        # Skip the hip of the hidden leg, since it is the same as the other one.
        if key == f"hip_{hidden_leg.lower()}":
            continue
        # Interpolate and synchronize the reconstruction and ArUco data.
        t, a_out, b_out = homogenize_3dpoint_data(
            reconstruction[key],
            aruco_processed[key]["data"],
            mcd_data["time"],
            aruco_processed[key]["time"],
        )
        trim_min = t > 1000
        trim_max = t < t.max() - 500
        trim = trim_min & trim_max
        # trim_t = t.copy()[trim]
        # Store the synchronized and trimmed data.
        homogeneous_data[key] = {
            "time": t[trim],
            "mcd": a_out[trim],
            "aruco": b_out[trim],
        }

    common_time = set()
    # Compute several error metrics for each joint.
    # print()
    for key in homogeneous_data.keys():
        # Euclidean error between the reconstruction and the ground truth.
        homogeneous_data[key]["euclidean"] = np.linalg.norm(
            homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"], axis=1
        )
        homogeneous_data[key]["euclidean_mean"] = np.mean(
            homogeneous_data[key]["euclidean"]
        )
        homogeneous_data[key]["mae"] = mean_absolute_error(
            homogeneous_data[key]["mcd"], homogeneous_data[key]["aruco"]
        )
        homogeneous_data[key]["mae_mean"] = np.mean(homogeneous_data[key]["mae"])
        # Variance and standard deviation of the Euclidean error.
        homogeneous_data[key]["eu_var"] = np.var(homogeneous_data[key]["euclidean"])
        homogeneous_data[key]["eu_std"] = np.std(homogeneous_data[key]["euclidean"])
        # Find the set of times common to all data points.
        if len(common_time) == 0:
            common_time = set(homogeneous_data[key]["time"])
        else:
            common_time = common_time.intersection(set(homogeneous_data[key]["time"]))

    # Compute the mean Euclidean error at each timestamp where all 5 points are present.
    common_time = sorted(list(common_time))
    five_points = np.empty((0, len(common_time)))
    for key in homogeneous_data.keys():
        # Filter the Euclidean errors to match the common times.
        isin = np.isin(homogeneous_data[key]["time"], list(common_time))
        try:
            five_points = np.append(
                five_points, [homogeneous_data[key]["euclidean"][isin]], axis=0
            )
        except Exception as e:
            print(key)
            print(e)
            print(files)
            pprint(five_points)
            pprint([homogeneous_data[key]["euclidean"][isin]])
            exit()

    # Prepare arrays to store the MCD and ArUco positions at the common times.
    mcd_common = np.empty((0, len(common_time), 3))
    aruco_common = np.empty((0, len(common_time), 3))
    for key in homogeneous_data.keys():
        isin = np.isin(homogeneous_data[key]["time"], common_time)
        mcd_common = np.append(mcd_common, [homogeneous_data[key]["mcd"][isin]], axis=0)
        aruco_common = np.append(
            aruco_common, [homogeneous_data[key]["aruco"][isin]], axis=0
        )

    # Compute the overall Euclidean error by averaging the positions of all points.
    gen_eucliden_error = np.empty((0))
    d_all_position = np.empty((0, 3))
    m_all_position = np.empty((0, 3))
    for i in range(len(common_time)):
        d = mcd_common[:, i, :]
        m = aruco_common[:, i, :]
        d_position = np.mean(d, axis=0)
        m_position = np.mean(m, axis=0)
        d_all_position = np.append(d_all_position, [d_position], axis=0)
        m_all_position = np.append(m_all_position, [m_position], axis=0)
        err = euclidean_error_3Dpoint(d_position, m_position)
        gen_eucliden_error = np.append(gen_eucliden_error, err)
    # Compute the Pearson correlation for each axis.
    # x_r, x_p_v = pearsonr(d_all_position[:, 0], m_all_position[:, 0])
    # y_r, y_p_v = pearsonr(d_all_position[:, 1], m_all_position[:, 1])
    # z_r, z_p_v = pearsonr(d_all_position[:, 2], m_all_position[:, 2])

    files[0].split(split_char)[-2]
    files[0].split(split_char)[-1].split(".")[0][-1]

    # """
    if use_3d:
        # 3D plot of the joint trajectories
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        # print(homogeneous_data.keys())
        for key in homogeneous_data.keys():
            if key in ["ankle_l", "knee_l", "hip_l"]:
                continue
            mcd = ax.plot(
                homogeneous_data[key]["mcd"][:, 0],
                homogeneous_data[key]["mcd"][:, 1],
                homogeneous_data[key]["mcd"][:, 2],
                label=f"MCD {key}",
                linewidth=1.5,
                color="b",
                alpha=0.5,
            )
            aruco = ax.plot(
                homogeneous_data[key]["aruco"][:, 0],
                homogeneous_data[key]["aruco"][:, 1],
                homogeneous_data[key]["aruco"][:, 2],
                label=f"aruco {key}",
                linewidth=1.5,
                color="r",
                alpha=0.5,
            )
        # Reconstructed pose
        reconstruction_pose = np.array(
            [
                reconstruction[key][-200]
                for key in ["ankle_l", "knee_l", "hip_l", "hip_r", "knee_r", "ankle_r"]
            ]
        )
        ax.plot(
            reconstruction_pose[:, 0],
            reconstruction_pose[:, 1],
            reconstruction_pose[:, 2],
            linewidth=3,
            color="b",
            marker="o",
            markersize=5,
            label="Reconstruction",
            alpha=0.6,
        )
        # Mean joint position
        # ax.plot(
        #     d_all_position[:, 0],
        #     d_all_position[:, 1],
        #     d_all_position[:, 2],
        #     label="MCD",
        #     color="blue",
        #     marker="o",
        #     markersize=2,
        #     alpha=0.5,
        # )
        # ax.plot(
        #     m_all_position[:, 0],
        #     m_all_position[:, 1],
        #     m_all_position[:, 2],
        #     label="arUco",
        #     color="red",
        #     marker="x",
        #     markersize=2,
        #     alpha=0.5,
        # )
        ax.legend([mcd[0], aruco[0]], ["Reconstruction", "Ground Truth"])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 1)
        ax.axis("equal")
        ax.set_title("Best")
        plt.show()
        fig.savefig(
            "Results/3d_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
    # """

    # plt.plot(sorted(common_time), gen_eucliden_error,
    #          label='General Euclidean Error')
    # plt.show()
    # print(f'5p-mean - {np.mean(five_points, axis=0).mean()}')
    # print(f'var - {np.var(np.mean(five_points, axis=0))}')
    # print(f'std - {np.std(np.mean(five_points, axis=0))}')
    # print(f'min - {np.mean(five_points, axis=0).min()}')
    # print(f'max - {np.mean(five_points, axis=0).max()}')

    # fig0, efp_axs = plt.subplots(5, 1, figsize=(10, 10))
    # fig1, efp_diff_axs = plt.subplots(5, 1, figsize=(10, 10))
    # fig2, comp_axis = plt.subplots(5, 1, figsize=(10, 10))
    # for i, key in enumerate(homogeneous_data.keys()):
    #     efp_axs[i].plot(homogeneous_data[key]['time'],
    #                     homogeneous_data[key]['euclidean'])
    #     efp_axs[i].set_title(f'Euclidean error {key}')
    #     efp_axs[i].set_xlabel('Time (ms)')
    #     efp_axs[i].set_ylabel('Euclidean error (m)')
    #     efp_axs[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #     efp_axs[i].set_ylim(0, 1)
    #     efp_axs[i].grid(1)

    #     efp_diff_axs[i].plot(homogeneous_data[key]['time'][1:],
    #                          np.diff(homogeneous_data[key]['euclidean']))
    #     efp_diff_axs[i].set_title(f'Euclidean error {key}')
    #     efp_diff_axs[i].set_xlabel('Time (ms)')
    #     efp_diff_axs[i].set_ylabel('Euclidean error (m)')
    #     efp_diff_axs[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #     efp_diff_axs[i].grid(1)

    #     for n in range(3):
    #         comp_axis[i].plot(homogeneous_data[key]['time'],
    #                           homogeneous_data[key]['mcd'][:, n]-homogeneous_data[key]['aruco'][:, n])
    #         comp_axis[i].set_title(f'{key} - {n}')
    #         comp_axis[i].set_xlabel('Time (ms)')
    #         comp_axis[i].set_ylabel('Position (m)')
    #         comp_axis[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #         comp_axis[i].set_ylim(-0.5, 0.5)
    #         comp_axis[i].grid(1)

    # Compute the mean absolute error per axis for the visible leg.
    axis_error = np.empty((0, 3))
    for key in homogeneous_data.keys():
        if key[-1] != hidden_leg.lower():
            # print(direction, key)
            continue
        axis_error = np.append(
            axis_error,
            [
                [
                    np.mean(
                        np.abs(
                            homogeneous_data[key]["mcd"][:, i]
                            - homogeneous_data[key]["aruco"][:, i]
                        )
                    )
                    for i in range(3)
                ]
            ],
            axis=0,
        )

    # print(np.mean(axis_error, axis=0))

    # --- Walking speed computation ---
    velocity_data = {}

    # Speed from the ArUco markers (ground truth)
    aruco_hip_key = "hip_r"
    if aruco_hip_key in aruco_processed:
        aruco_hip_t = aruco_processed[aruco_hip_key]["time"].astype(float)
        aruco_hip_pos = aruco_processed[aruco_hip_key]["data"]
        velocity_data["aruco_hip"] = compute_displacement_velocity(
            aruco_hip_pos, aruco_hip_t, trim_ms=1000
        )

    rec_hip_pos = (reconstruction["hip_r"] + reconstruction["hip_l"]) / 2.0
    velocity_data["reconstruction_hip"] = compute_displacement_velocity(
        rec_hip_pos, mcd_data["time"], trim_ms=1000
    )

    for side in ["r", "l"]:
        ankle_key = f"ankle_{side}"
        if ankle_key in aruco_processed:
            a_t = aruco_processed[ankle_key]["time"].astype(float)
            a_pos = aruco_processed[ankle_key]["data"]
            velocity_data[f"aruco_{ankle_key}"] = compute_displacement_velocity(
                a_pos, a_t, trim_ms=1000
            )
        velocity_data[f"reconstruction_{ankle_key}"] = compute_displacement_velocity(
            reconstruction[ankle_key], mcd_data["time"], trim_ms=1000
        )

    # =====================================================================
    # ADDITIONAL SCIENTIFIC METRICS
    # =====================================================================
    extra_metrics = {}

    # --- 1. RMSE per joint ---
    rmse_per_joint = {}
    for key in homogeneous_data.keys():
        diff = homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"]
        rmse = np.sqrt(np.mean(diff**2))
        rmse_per_joint[key] = rmse
    extra_metrics["rmse_per_joint"] = rmse_per_joint
    extra_metrics["rmse_mean"] = np.mean(list(rmse_per_joint.values()))

    # --- 2. Pearson correlation per axis ---
    pearson_per_joint = {}
    axis_labels = ["X", "Y", "Z"]
    for key in homogeneous_data.keys():
        joint_corr = {}
        for ax_i, ax_name in enumerate(axis_labels):
            mcd_ax = homogeneous_data[key]["mcd"][:, ax_i]
            aruco_ax = homogeneous_data[key]["aruco"][:, ax_i]
            if np.std(mcd_ax) > 1e-10 and np.std(aruco_ax) > 1e-10:
                r_val, p_val = pearsonr(mcd_ax, aruco_ax)
            else:
                r_val, p_val = 0.0, 1.0
            joint_corr[ax_name] = {"r": r_val, "p_value": p_val}
        pearson_per_joint[key] = joint_corr
    extra_metrics["pearson_per_joint"] = pearson_per_joint

    # --- 3. Bland-Altman data (overall mean position) ---
    bland_altman = {}
    for ax_i, ax_name in enumerate(axis_labels):
        all_mcd_ax = np.concatenate(
            [homogeneous_data[key]["mcd"][:, ax_i] for key in homogeneous_data.keys()]
        )
        all_aruco_ax = np.concatenate(
            [homogeneous_data[key]["aruco"][:, ax_i] for key in homogeneous_data.keys()]
        )
        mean_vals = (all_mcd_ax + all_aruco_ax) / 2.0
        diff_vals = all_mcd_ax - all_aruco_ax
        bland_altman[ax_name] = {
            "mean": mean_vals,
            "diff": diff_vals,
            "bias": np.mean(diff_vals),
            "std_diff": np.std(diff_vals),
            "upper_loa": np.mean(diff_vals) + 1.96 * np.std(diff_vals),
            "lower_loa": np.mean(diff_vals) - 1.96 * np.std(diff_vals),
        }
    extra_metrics["bland_altman"] = bland_altman

    # --- 4. Stride length ---
    stride_data = {}
    for side in ["r", "l"]:
        ankle_key = f"ankle_{side}"
        # From the reconstruction
        rec_strides, rec_stride_t = detect_stride_lengths(
            reconstruction[ankle_key], mcd_data["time"]
        )
        stride_data[f"reconstruction_{ankle_key}"] = {
            "lengths": rec_strides,
            "times": rec_stride_t,
            "mean": np.mean(rec_strides) if len(rec_strides) > 0 else 0.0,
            "std": np.std(rec_strides) if len(rec_strides) > 0 else 0.0,
        }
        # From ArUco
        if ankle_key in aruco_processed:
            aruco_ankle_pos = aruco_processed[ankle_key]["data"]
            aruco_ankle_t = aruco_processed[ankle_key]["time"].astype(float)
            aruco_strides, aruco_stride_t = detect_stride_lengths(
                aruco_ankle_pos, aruco_ankle_t
            )
            stride_data[f"aruco_{ankle_key}"] = {
                "lengths": aruco_strides,
                "times": aruco_stride_t,
                "mean": np.mean(aruco_strides) if len(aruco_strides) > 0 else 0.0,
                "std": np.std(aruco_strides) if len(aruco_strides) > 0 else 0.0,
            }
    extra_metrics["stride_data"] = stride_data

    # --- 5. Knee ROM (Range of Motion) ---
    knee_rom = {}
    for side in ["r", "l"]:
        hip_key = f"hip_{side}"
        knee_key = f"knee_{side}"
        ankle_key = f"ankle_{side}"

        # ROM from the reconstruction
        rec_knee_angles = compute_knee_angles_series(
            reconstruction[hip_key], reconstruction[knee_key], reconstruction[ankle_key]
        )
        knee_rom[f"reconstruction_{side}"] = {
            "angles": rec_knee_angles,
            "rom": np.max(rec_knee_angles) - np.min(rec_knee_angles),
            "max": np.max(rec_knee_angles),
            "min": np.min(rec_knee_angles),
            "mean": np.mean(rec_knee_angles),
        }

        # ROM from ArUco (if synchronized data are available)
        if knee_key in homogeneous_data:
            aruco_knee_angles = compute_knee_angles_series(
                (
                    homogeneous_data[hip_key]["aruco"]
                    if hip_key in homogeneous_data
                    else homogeneous_data[f"hip_{direction.lower()}"]["aruco"]
                ),
                homogeneous_data[knee_key]["aruco"],
                homogeneous_data[ankle_key]["aruco"],
            )
            mcd_knee_angles = compute_knee_angles_series(
                (
                    homogeneous_data[hip_key]["mcd"]
                    if hip_key in homogeneous_data
                    else homogeneous_data[f"hip_{direction.lower()}"]["mcd"]
                ),
                homogeneous_data[knee_key]["mcd"],
                homogeneous_data[ankle_key]["mcd"],
            )
            knee_rom[f"aruco_{side}"] = {
                "angles": aruco_knee_angles,
                "rom": np.max(aruco_knee_angles) - np.min(aruco_knee_angles),
                "max": np.max(aruco_knee_angles),
                "min": np.min(aruco_knee_angles),
                "mean": np.mean(aruco_knee_angles),
            }
            knee_rom[f"homogeneous_mcd_{side}"] = {
                "angles": mcd_knee_angles,
                "rom": np.max(mcd_knee_angles) - np.min(mcd_knee_angles),
                "max": np.max(mcd_knee_angles),
                "min": np.min(mcd_knee_angles),
                "mean": np.mean(mcd_knee_angles),
            }
    extra_metrics["knee_rom"] = knee_rom

    # Return the processing results.
    return (
        gen_eucliden_error,
        sorted(list(common_time)),
        reconstruction,
        aruco_processed,
        homogeneous_data,
        velocity_data,
        extra_metrics,
    )


def animation(rec, aruco, title):
    fig, ax = plt.figure(), plt.axes(projection="3d")
    fig.subplots_adjust(left=0, right=0.99, top=0.98, bottom=0.03, hspace=0, wspace=0.2)
    mcd = ax.plot([], [], [], "o-", label="MCD", color="blue")[0]
    aru = ax.plot([], [], [], "o-", label="arUco", color="red")[0]
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.suptitle(title)
    ax.legend()
    fig.show()
    i = 0

    aruco_inter = {k: interpolate_point(v["time"], v["data"]) for k, v in aruco.items()}
    # print(aruco_inter.keys())
    while plt.fignum_exists(fig.number):
        try:
            aruco_pose = np.empty((0, 3))
            for k in ["ankle_r", "knee_r", "hip_r", "hip_l", "knee_l", "ankle_l"]:
                try:
                    aruco_pose = np.append(
                        aruco_pose,
                        [[inter(i * 10) for inter in aruco_inter[k]]],
                        axis=0,
                    )
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass

            pose = np.empty((0, 3))
            for k in ["ankle_r", "knee_r", "hip_r", "hip_l", "knee_l", "ankle_l"]:
                pose = np.append(pose, [rec[k][i]], axis=0)
            mcd.set_data_3d(pose[:, 0], pose[:, 1], pose[:, 2])
            aru.set_data_3d(aruco_pose[:, 0], aruco_pose[:, 1], aruco_pose[:, 2])
            ax.set_xlim(np.mean(pose[3:5, 0]) - 0.5, np.mean(pose[3:5, 0]) + 0.5)
            ax.set_ylim(np.mean(pose[3:5, 1]) - 0.5, np.mean(pose[3:5, 1]) + 0.5)
            # ax.set_zlim(np.mean(pose[3:5, 0])-.5, np.mean(pose[3:5, 0])+.5)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

            i += 1
            if i >= len(rec["hip_r"]):
                i = 0
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Analytical IMU gait pipeline over the whole dataset")
    _parser.add_argument(
        "--dataset",
        default="DataSet",
        help="Dataset folder containing one sub-folder per subject (default: DataSet/ in the repository root)",
    )
    files = get_dataset(_parser.parse_args().dataset)
    if len(files) == 0:
        print("No files selected")
        exit(1)

    target_files = [f for f in files if f[0].split(split_char)[-2] != "0"]

    print(f"\n  📂 Processing {len(target_files)} captures with multiprocessing...")
    t_start = time.time()

    with Pool() as pool:
        raw_results = pool.map(process_files, target_files)

    elapsed = time.time() - t_start

    valid_data = [
        (f, res) for f, res in zip(target_files, raw_results) if res is not None
    ]
    target_files = [f for f, res in valid_data]
    gen_errors = [res for f, res in valid_data]

    mean_error_per_point = np.empty(0)
    for file in gen_errors:
        homogeneous = file[4]
        mean_error_per_point = np.append(
            mean_error_per_point,
            np.mean([homogeneous[key]["euclidean_mean"] for key in homogeneous.keys()]),
        )

    gen_mean_error = np.array([np.mean(err[0]) for err in gen_errors])

    best_idx = np.argmin(gen_mean_error)
    worst_idx = np.argmax(gen_mean_error)
    mean_idx = np.argmin(np.abs(gen_mean_error - np.mean(gen_mean_error)))

    # ─────────────────────────────────────────────────────────────────────
    #  PRINT RESULTS
    # ─────────────────────────────────────────────────────────────────────

    W = 72  # box width

    print()
    print("═" * W)
    print("  DATASET SUMMARY")
    print("═" * W)
    print(
        f"  Processed captures:        {len(gen_errors)} / {len(target_files) + (len(raw_results) - len(gen_errors))}"
    )
    print(f"  Valid captures:            {len(gen_errors)}")
    print(f"  Processing time:           {elapsed:.1f} s")
    print("═" * W)

    # ── 1. Overall Euclidean error ───────────────────────────────────────
    print()
    print("═" * W)
    print("  📊 OVERALL EUCLIDEAN ERROR (centroid of 5 joints)")
    print("═" * W)
    print(f"  Mean error:                {np.mean(gen_mean_error):.6f} m")
    print(f"  Standard deviation:        {np.std(gen_mean_error):.6f} m")
    print(f"  Variance:                  {np.var(gen_mean_error):.6f} m²")
    print(f"  Minimum:                   {np.min(gen_mean_error):.6f} m")
    print(f"  Maximum:                   {np.max(gen_mean_error):.6f} m")
    print(
        f"  Range:                     {np.max(gen_mean_error) - np.min(gen_mean_error):.6f} m"
    )
    print("─" * W)
    print(f"  📐 Mean error per point (average over joints):")
    print(f"  Mean:                      {np.mean(mean_error_per_point):.6f} m")
    print(f"  Standard deviation:        {np.std(mean_error_per_point):.6f} m")
    print(f"  Minimum:                   {np.min(mean_error_per_point):.6f} m")
    print(f"  Maximum:                   {np.max(mean_error_per_point):.6f} m")
    print("═" * W)

    # ── 2. Notable captures ──────────────────────────────────────────────
    print()
    print("═" * W)
    print("  🏆 NOTABLE CAPTURES")
    print("═" * W)
    for label, idx, icon in [
        ("Best", best_idx, "🟢"),
        ("Worst", worst_idx, "🔴"),
        ("Mean", mean_idx, "🔵"),
    ]:
        file_label = "/".join(target_files[idx][0].split(split_char)[-2:])
        print(
            f"  {icon} {label + ':':<12} {file_label:<30} error = {gen_mean_error[idx]:.6f} m"
        )
    print("═" * W)

    # ── 3. Error per subject ─────────────────────────────────────────────
    errors_by_subject = {}
    for f, err in zip(target_files, gen_errors):
        subject = f[0].split(split_char)[-2]
        if subject not in errors_by_subject:
            errors_by_subject[subject] = []
        errors_by_subject[subject].extend(err[0])

    subjects = sorted(
        errors_by_subject.keys(), key=lambda x: int(x) if x.isdigit() else x
    )

    print()
    print("═" * W)
    print("  📋 ERROR PER SUBJECT")
    print("═" * W)
    print(
        f"  {'Subject':>8}  {'Mean (m)':>12}  {'Std (m)':>12}  {'Min (m)':>12}  {'Max (m)':>12}"
    )
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for s in subjects:
        vals = np.array(errors_by_subject[s])
        print(
            f"  {s:>8}  {np.mean(vals):>12.6f}  {np.std(vals):>12.6f}  {np.min(vals):>12.6f}  {np.max(vals):>12.6f}"
        )
    print("═" * W)

    # ── 4. Walking speed ─────────────────────────────────────────────────
    rec_hip_velocities = []
    aruco_hip_velocities = []
    for err in gen_errors:
        vel_data = err[5]
        rec_hip_velocities.append(
            vel_data.get("reconstruction_hip", {}).get("mean_velocity", 0.0)
        )
        aruco_hip_velocities.append(
            vel_data.get("aruco_hip", {}).get("mean_velocity", 0.0)
        )
    rec_hip_velocities = np.array(rec_hip_velocities)
    aruco_hip_velocities = np.array(aruco_hip_velocities)

    print()
    print("═" * W)
    print("  🚶 WALKING SPEED (hip)")
    print("═" * W)
    print(f"  {'':30}  {'Reconstruction':>16}  {'ArUco (GT)':>16}")
    print(f"  {'─'*30}  {'─'*16}  {'─'*16}")
    print(
        f"  {'Mean speed (m/s)':30}  {np.mean(rec_hip_velocities):>16.4f}  {np.mean(aruco_hip_velocities):>16.4f}"
    )
    print(
        f"  {'± Std. deviation (m/s)':30}  {np.std(rec_hip_velocities):>16.4f}  {np.std(aruco_hip_velocities):>16.4f}"
    )
    print(
        f"  {'Minimum (m/s)':30}  {np.min(rec_hip_velocities):>16.4f}  {np.min(aruco_hip_velocities):>16.4f}"
    )
    print(
        f"  {'Maximum (m/s)':30}  {np.max(rec_hip_velocities):>16.4f}  {np.max(aruco_hip_velocities):>16.4f}"
    )

    # Speed per subject
    vel_by_subject_rec = {}
    vel_by_subject_aruco = {}
    for i, f in enumerate(target_files):
        subject = f[0].split(split_char)[-2]
        if subject not in vel_by_subject_rec:
            vel_by_subject_rec[subject] = []
            vel_by_subject_aruco[subject] = []
        vel_by_subject_rec[subject].append(rec_hip_velocities[i])
        vel_by_subject_aruco[subject].append(aruco_hip_velocities[i])

    print()
    print(f"  Mean speed per subject:")
    print(f"  {'Subject':>8}  {'Rec. (m/s)':>14}  {'ArUco (m/s)':>14}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}")
    for s in sorted(
        vel_by_subject_rec.keys(), key=lambda x: int(x) if x.isdigit() else x
    ):
        print(
            f"  {s:>8}  {np.mean(vel_by_subject_rec[s]):>14.4f}  {np.mean(vel_by_subject_aruco[s]):>14.4f}"
        )
    print("═" * W)

    # ── 5. Scientific metrics ────────────────────────────────────────────
    all_rmse_means = [err[6]["rmse_mean"] for err in gen_errors]
    all_rmse_per_joint = {}
    for err in gen_errors:
        for joint, rmse_val in err[6]["rmse_per_joint"].items():
            if joint not in all_rmse_per_joint:
                all_rmse_per_joint[joint] = []
            all_rmse_per_joint[joint].append(rmse_val)

    axis_labels = ["X", "Y", "Z"]
    pearson_all = {ax: {"r_vals": [], "p_vals": []} for ax in axis_labels}
    pearson_by_joint = {}
    for err in gen_errors:
        for joint, corr_data in err[6]["pearson_per_joint"].items():
            if joint not in pearson_by_joint:
                pearson_by_joint[joint] = {ax: [] for ax in axis_labels}
            for ax in axis_labels:
                r_val = corr_data[ax]["r"]
                pearson_by_joint[joint][ax].append(r_val)
                pearson_all[ax]["r_vals"].append(r_val)
                pearson_all[ax]["p_vals"].append(corr_data[ax]["p_value"])

    print()
    print("═" * W)
    print("  🔬 SCIENTIFIC METRICS")
    print("═" * W)

    # 5a. RMSE
    print()
    print(f"  1) RMSE (Root Mean Square Error)")
    print(f"  {'─'*40}")
    print(
        f"     RMSE general:  {np.mean(all_rmse_means):.4f} ± {np.std(all_rmse_means):.4f} m"
    )
    print()
    print(f"     {'Joint':<12}  {'RMSE (m)':>12}")
    print(f"     {'─'*12}  {'─'*12}")
    for joint in sorted(all_rmse_per_joint.keys()):
        vals = np.array(all_rmse_per_joint[joint])
        print(f"     {joint:<12}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # 5b. Pearson
    print()
    print(f"  2) Pearson correlation")
    print(f"  {'─'*40}")
    print(f"     Overall mean:")
    for ax in axis_labels:
        r_mean = np.mean(pearson_all[ax]["r_vals"])
        r_std = np.std(pearson_all[ax]["r_vals"])
        print(f"       Axis {ax}:  r = {r_mean:.4f} ± {r_std:.4f}")
    print()
    print(f"     {'Joint':<12}  {'r_X':>8}  {'r_Y':>8}  {'r_Z':>8}")
    print(f"     {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
    for joint in sorted(pearson_by_joint.keys()):
        rx = np.mean(pearson_by_joint[joint]["X"])
        ry = np.mean(pearson_by_joint[joint]["Y"])
        rz = np.mean(pearson_by_joint[joint]["Z"])
        print(f"     {joint:<12}  {rx:>8.4f}  {ry:>8.4f}  {rz:>8.4f}")

    # 5c. Bland-Altman
    ba_data = gen_errors[mean_idx][6]["bland_altman"]
    ba_agg = {ax: {"bias": [], "upper": [], "lower": []} for ax in axis_labels}
    for err in gen_errors:
        for ax in axis_labels:
            ba = err[6]["bland_altman"][ax]
            ba_agg[ax]["bias"].append(ba["bias"])
            ba_agg[ax]["upper"].append(ba["upper_loa"])
            ba_agg[ax]["lower"].append(ba["lower_loa"])

    print()
    print(f"  3) Bland-Altman")
    print(f"  {'─'*40}")
    print(f"     Mean capture:")
    for ax in axis_labels:
        ba = ba_data[ax]
        print(
            f"       Axis {ax}:  Bias = {ba['bias']:.4f} m,  LoA = [{ba['lower_loa']:.4f}, {ba['upper_loa']:.4f}] m"
        )
    print(f"     Aggregate (all captures):")
    for ax in axis_labels:
        print(
            f"       Axis {ax}:  Bias = {np.mean(ba_agg[ax]['bias']):.4f} ± {np.std(ba_agg[ax]['bias']):.4f} m"
        )

    # 5d. Stride length
    rec_stride_all = []
    aruco_stride_all = []
    for err in gen_errors:
        sd = err[6]["stride_data"]
        for side in ["r", "l"]:
            rec_key = f"reconstruction_ankle_{side}"
            aruco_key = f"aruco_ankle_{side}"
            if rec_key in sd and len(sd[rec_key]["lengths"]) > 0:
                rec_stride_all.extend(sd[rec_key]["lengths"])
            if aruco_key in sd and len(sd[aruco_key]["lengths"]) > 0:
                aruco_stride_all.extend(sd[aruco_key]["lengths"])
    rec_stride_all = np.array(rec_stride_all)
    aruco_stride_all = np.array(aruco_stride_all)

    print()
    print(f"  4) Stride length")
    print(f"  {'─'*40}")
    if len(rec_stride_all) > 0:
        print(
            f"     Reconstruction:  {np.mean(rec_stride_all):.4f} ± {np.std(rec_stride_all):.4f} m  (n={len(rec_stride_all)})"
        )
    else:
        print(f"     Reconstruction:  No strides detected")
    if len(aruco_stride_all) > 0:
        print(
            f"     ArUco (GT):      {np.mean(aruco_stride_all):.4f} ± {np.std(aruco_stride_all):.4f} m  (n={len(aruco_stride_all)})"
        )
    else:
        print(f"     ArUco (GT):      No strides detected")

    # 5e. Knee ROM
    rom_rec_all = []
    rom_aruco_all = []
    for err in gen_errors:
        kr = err[6]["knee_rom"]
        for side in ["r", "l"]:
            if f"reconstruction_{side}" in kr:
                rom_rec_all.append(kr[f"reconstruction_{side}"]["rom"])
            if f"aruco_{side}" in kr:
                rom_aruco_all.append(kr[f"aruco_{side}"]["rom"])
    rom_rec_all = np.array(rom_rec_all) if rom_rec_all else np.array([0.0])
    rom_aruco_all = np.array(rom_aruco_all) if rom_aruco_all else np.array([0.0])

    print()
    print(f"  5) Knee ROM (Range of Motion)")
    print(f"  {'─'*40}")
    print(
        f"     Reconstruction:  {np.mean(rom_rec_all):.2f} ± {np.std(rom_rec_all):.2f}°"
    )
    print(
        f"     ArUco (GT):      {np.mean(rom_aruco_all):.2f} ± {np.std(rom_aruco_all):.2f}°"
    )
    if len(rom_rec_all) > 0 and len(rom_aruco_all) > 0:
        rom_diff = np.mean(rom_rec_all) - np.mean(rom_aruco_all)
        print(f"     Difference:      {rom_diff:+.2f}°")
    print()
    print("═" * W)

    # ─────────────────────────────────────────────────────────────────────
    #  PLOTS
    # ─────────────────────────────────────────────────────────────────────

    print()
    print("  🎨 Generating plots...")

    # Error over time: best, worst, mean
    fig_best, ax_best = plt.subplots(1, 1, figsize=figsize)
    ax_best.plot(
        gen_errors[best_idx][1],
        gen_errors[best_idx][0],
        label="Best error",
        color="green",
    )
    ax_best.set_title("Best error")
    ax_best.set_xlabel("Time (ms)")
    ax_best.set_ylabel("Euclidean error (m)")
    ax_best.grid(True)
    ax_best.set_ylim(0, 1)

    fig_worst, ax_worst = plt.subplots(1, 1, figsize=figsize)
    ax_worst.plot(
        gen_errors[worst_idx][1],
        gen_errors[worst_idx][0],
        label="Worst error",
        color="red",
    )
    ax_worst.set_title("Worst error")
    ax_worst.set_xlabel("Time (ms)")
    ax_worst.set_ylabel("Euclidean error (m)")
    ax_worst.grid(True)
    ax_worst.set_ylim(0, 1)

    fig_mean, ax_mean = plt.subplots(1, 1, figsize=figsize)
    ax_mean.plot(
        gen_errors[mean_idx][1],
        gen_errors[mean_idx][0],
        label="Mean error",
        color="blue",
    )
    ax_mean.set_title("Average error")
    ax_mean.set_xlabel("Time (ms)")
    ax_mean.set_ylabel("Euclidean error (m)")
    ax_mean.grid(True)
    ax_mean.set_ylim(0, 1)

    # All errors overlaid
    all_fig, all_ax = plt.subplots(1, 1, figsize=figsize)
    for i, (err, t, *_) in enumerate(gen_errors):
        all_ax.plot(t, err, label=f"File {i+1}")
    all_ax.set_title("All errors")
    all_ax.set_xlabel("Time (ms)")
    all_ax.set_ylabel("Euclidean error (m)")
    all_ax.grid(True)
    all_ax.set_ylim(0, 1)

    # Boxplot per subject
    data_to_plot = [errors_by_subject[s] for s in subjects]
    fig_box, ax_box = plt.subplots(figsize=figsize)
    ax_box.boxplot(data_to_plot, labels=subjects)
    ax_box.set_title("Euclidean Error Distribution by Subject")
    ax_box.set_xlabel("Subject / Volunteer")
    ax_box.set_ylabel("Euclidean Error (m)")
    ax_box.grid(True)
    ax_box.set_ylim(0, 0.8)
    fig_box.savefig("Results/boxplot_error.png", dpi=300, bbox_inches="tight")
    print("  ✓ Results/boxplot_error.png")

    # Speed: best, worst, mean, all
    for tag, idx in [("best", best_idx), ("worst", worst_idx), ("mean", mean_idx)]:
        vel = gen_errors[idx][5]
        fig_v, ax_v = plt.subplots(1, 1, figsize=figsize)
        if (
            "reconstruction_hip" in vel
            and len(vel["reconstruction_hip"]["velocities"]) > 0
        ):
            ax_v.plot(
                vel["reconstruction_hip"]["timestamps"] / 1000.0,
                vel["reconstruction_hip"]["velocities"],
                label="Reconstruction (hip)",
                color="blue",
                alpha=0.7,
                linewidth=0.8,
            )
        if "aruco_hip" in vel and len(vel["aruco_hip"]["velocities"]) > 0:
            ax_v.plot(
                vel["aruco_hip"]["timestamps"] / 1000.0,
                vel["aruco_hip"]["velocities"],
                label="ArUco (hip)",
                color="red",
                alpha=0.7,
                linewidth=0.8,
            )
        ax_v.set_title(f"Walking speed – {tag.capitalize()}")
        ax_v.set_xlabel("Time (s)")
        ax_v.set_ylabel("Speed (m/s)")
        ax_v.legend()
        ax_v.grid(True)
        ax_v.set_ylim(0, 3)
        fig_v.savefig(f"Results/velocity_{tag}.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Results/velocity_{tag}.png")

    fig_vel_all, ax_vel_all = plt.subplots(1, 1, figsize=figsize)
    for err in gen_errors:
        vd = err[5]
        if (
            "reconstruction_hip" in vd
            and len(vd["reconstruction_hip"]["velocities"]) > 0
        ):
            ax_vel_all.plot(
                vd["reconstruction_hip"]["timestamps"] / 1000.0,
                vd["reconstruction_hip"]["velocities"],
                alpha=0.3,
                linewidth=0.6,
            )
    ax_vel_all.set_title("Walking speed – All captures")
    ax_vel_all.set_xlabel("Time (s)")
    ax_vel_all.set_ylabel("Speed (m/s)")
    ax_vel_all.grid(True)
    ax_vel_all.set_ylim(0, 3)
    fig_vel_all.savefig("Results/velocity_all.png", dpi=300, bbox_inches="tight")
    print("  ✓ Results/velocity_all.png")

    # Bland-Altman
    fig_ba, axs_ba = plt.subplots(1, 3, figsize=(15, 5))
    fig_ba.suptitle("Bland-Altman: Reconstruction vs Ground Truth (mean capture)")
    for ax_i, ax_name in enumerate(axis_labels):
        ba = ba_data[ax_name]
        axs_ba[ax_i].scatter(ba["mean"], ba["diff"], alpha=0.15, s=3, color="steelblue")
        axs_ba[ax_i].axhline(
            ba["bias"],
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Bias: {ba['bias']:.4f} m",
        )
        axs_ba[ax_i].axhline(
            ba["upper_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"+1.96σ: {ba['upper_loa']:.4f} m",
        )
        axs_ba[ax_i].axhline(
            ba["lower_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"-1.96σ: {ba['lower_loa']:.4f} m",
        )
        axs_ba[ax_i].set_xlabel(f"Mean {ax_name} (m)")
        axs_ba[ax_i].set_ylabel(f"Difference {ax_name} (m)")
        axs_ba[ax_i].set_title(f"Axis {ax_name}")
        axs_ba[ax_i].legend(fontsize=7)
        axs_ba[ax_i].grid(True, alpha=0.3)
    fig_ba.tight_layout()
    fig_ba.savefig("Results/bland_altman.png", dpi=300, bbox_inches="tight")
    print("  ✓ Results/bland_altman.png")

    # Knee ROM
    kr_mean = gen_errors[mean_idx][6]["knee_rom"]
    fig_rom, axs_rom = plt.subplots(1, 2, figsize=(12, 5))
    fig_rom.suptitle("Knee angle – Mean capture")
    for s_i, side in enumerate(["r", "l"]):
        side_label = "Right" if side == "r" else "Left"
        rec_key = f"reconstruction_{side}"
        aruco_key = f"aruco_{side}"
        hom_key = f"homogeneous_mcd_{side}"
        if hom_key in kr_mean and aruco_key in kr_mean:
            axs_rom[s_i].plot(
                kr_mean[hom_key]["angles"],
                label="Reconstruction",
                color="blue",
                alpha=0.8,
            )
            axs_rom[s_i].plot(
                kr_mean[aruco_key]["angles"], label="ArUco (GT)", color="red", alpha=0.8
            )
        elif rec_key in kr_mean:
            axs_rom[s_i].plot(
                kr_mean[rec_key]["angles"],
                label="Reconstruction",
                color="blue",
                alpha=0.8,
            )
        axs_rom[s_i].set_title(f"{side_label} knee")
        axs_rom[s_i].set_xlabel("Sample")
        axs_rom[s_i].set_ylabel("Flexion angle (°)")
        axs_rom[s_i].legend()
        axs_rom[s_i].grid(True, alpha=0.3)
    fig_rom.tight_layout()
    fig_rom.savefig("Results/knee_rom.png", dpi=300, bbox_inches="tight")
    print("  ✓ Results/knee_rom.png")

    print()
    print("═" * W)
    print("  ✅ Processing complete.")
    print("═" * W)

    process_files(target_files[best_idx], use_3d=True)

    animation(gen_errors[mean_idx][2], gen_errors[mean_idx][3], "mean")
