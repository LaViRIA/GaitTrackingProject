from ahrs.filters import EKF, Madgwick
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation as R
import numpy as np
from pprint import pprint
import csv
import glob
import sys
from scipy.signal import butter, filtfilt, lfilter, detrend, find_peaks
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from pathlib import Path

# from sklearn.linear_model import LinearRegression
import time
import os
from multiprocessing import Pool

# ==============================================================================
# Configuración óptima obtenida de la matriz de experimentación
# (Results/experiment_matrix/results.txt)
# ==============================================================================
OPTIMAL_CONFIG = {
    "cutoff_freq": 9,  # Hz (Frecuencia de corte Butterworth)
    "filter_order": 6,  # Orden del filtro Butterworth
    "pca_percentile": 20,  # Percentil PCA para corrección funcional de yaw
    "madgwick_gain": 0.02,  # Ganancia del filtro Madgwick (beta)
    "slerp_clip_min": 0.3,  # Clamp mín. Slerp cadera
    "slerp_clip_max": 0.7,  # Clamp máx. Slerp cadera (1.0 - slerp_clip_min)
    "calibration_samples": 100,  # Muestras estáticas de calibración inicial
    "fusion_filter": "madgwick",  # Algoritmo de fusión sensorial
    "pivot_height_threshold": 0.005,  # Umbral de altura de pivote (m)
    "swing_velocity_threshold": 0.3,  # Umbral de velocidad angular de swing (rad/s)
}

cache_dir = matplotlib.get_cachedir()
font_cache_dir = os.path.join(cache_dir, "fontlist-v310.json")
if os.path.exists(font_cache_dir):
    os.remove(font_cache_dir)
    print("Font cache removed.")
else:
    print("Font cache not found.")
figsize = (7, 6)
title = ""

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
if sys.platform == "linux":
    split_char = "/"
else:
    split_char = "\\"


def distancia_punto_segmento(P, A, B):
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    # Vectores AB y AP
    AB = B - A
    AP = P - A
    # Producto punto
    dot_AB_AB = np.dot(AB, AB)
    dot_AP_AB = np.dot(AP, AB)

    # Factor de proyección
    t = dot_AP_AB / dot_AB_AB
    # Si la proyección está dentro del segmento
    if 0 <= t <= 1:
        # Distancia perpendicular (proyección sobre el segmento)
        proyeccion = A + t * AB
        distancia = np.linalg.norm(P - proyeccion)
        # print("if ", end="")
    else:
        # Si no, la distancia es al extremo más cercano
        dist_A = np.linalg.norm(P - A)
        dist_B = np.linalg.norm(P - B)
        distancia = min(dist_A, dist_B)
    # print(distancia)
    return distancia


def distancia_punto_forma(p, forma, cerrada=True):
    # print("- ", p)
    distances = []
    for i in range(len(forma)):
        if i == 0 and not cerrada:
            continue
        distances.append(distancia_punto_segmento(p, forma[i - 1], forma[i]))
    return min(distances)


def distancia_dos_puntos(p1, p2):
    return np.linalg.norm(p1 - p2)


def distancia_punto_forma2(p, forma):
    return min([distancia_dos_puntos(p, f) for f in forma])


def conjugado(q):
    """
    Calcula el conjugado de un cuaternión.

    Args:
        q (np.array): Cuaternión de entrada en formato [w, x, y, z].

    Returns:
        np.array: Cuaternión conjugado.
    """
    return q * np.array([1, -1, -1, -1])


def multiplicacion(q1, q2):
    """
    Multiplica dos cuaterniones (q1 * q2).

    Args:
        q1 (np.array): Primer cuaternión [w1, x1, y1, z1].
        q2 (np.array): Segundo cuaternión [w2, x2, y2, z2].

    Returns:
        np.array: Cuaternión resultante.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def rotacion_relativa(q1, q2):
    """
    Calcula la rotación relativa para ir de la orientación q1 a la q2.

    La fórmula es: q_rel = conjugado(q1) * q2

    Args:
        q1 (np.array): Cuaternión de la orientación inicial [w, x, y, z].
        q2 (np.array): Cuaternión de la orientación final [w, x, y, z].

    Returns:
        np.array: Cuaternión que representa la rotación relativa.
    """
    # Asegurarse de que los cuaterniones estén normalizados (opcional pero recomendado)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calcular el conjugado de la orientación inicial
    q1_conj = conjugado(q1)

    # Multiplicar el conjugado de q1 por q2
    q_rel = multiplicacion(q1_conj, q2)

    return q_rel


def rotation_matrix_from_vectors(vec_sensor, vec_world=[0, 0, 1]):
    """Calcula la matriz de rotación para alinear vec_sensor con vec_world."""
    vec_sensor = vec_sensor / np.linalg.norm(vec_sensor)
    vec_world = vec_world / np.linalg.norm(vec_world)

    v = np.cross(vec_sensor, vec_world)
    c = np.dot(vec_sensor, vec_world)
    s = np.linalg.norm(v)

    # Matriz de rotación (Fórmula de Rodrigues)
    if s < 1e-6:  # Vectores paralelos
        return np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def low_pass_filter(data, cutoff_freq, fs, order=OPTIMAL_CONFIG["filter_order"]):
    """
    Applies a low-pass filter to the given data.

    Args:
        data (array-like): Array of values corresponding to the timestamps.
        cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling frequency.
        order (int): Filter order (default: 6).

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


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica un filtro pasa-bandas Butterworth a una señal.

    Parámetros:
    data (array_like): La señal de entrada a filtrar.
    lowcut (float): La frecuencia de corte inferior del filtro (en Hz).
    highcut (float): La frecuencia de corte superior del filtro (en Hz).
    fs (float): La frecuencia de muestreo de la señal (en Hz).
    order (int, opcional): El orden del filtro Butterworth. Por defecto es 5.

    Retorna:
    array_like: La señal filtrada.
    """
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


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


def get_dataset() -> list[tuple[str, str]]:
    """
    Get the dataset folder and return a list of tuples containing the paths to the mcd and arUco files.
    """

    dataset_path = filedialog.askdirectory(
        title="Select the dataset folder",
        initialdir="/media/moybadajoz/Nuevo vol/DataSet With Vid",
    )
    print(dataset_path)
    if len(dataset_path) == 0 or dataset_path == "":
        print("No folder selected")
        return []
    subjects = glob.glob("*", root_dir=dataset_path)
    files = []
    for s in subjects:
        for i in range(4):
            mcd_file = f"{dataset_path}{split_char}{s}{split_char}mcd-{i}.csv"
            aruco_file = f"{dataset_path}{split_char}{s}{split_char}arUcos-{i}.csv"
            if not Path(mcd_file).exists() or not Path(aruco_file).exists():
                continue
            files.append((mcd_file, aruco_file))
    return files


def get_dataset2():
    """
    Get the dataset folder and return a list of tuples containing the paths to the mcd and arUco files.
    """

    dataset_path = filedialog.askdirectory(
        title="Select the dataset folder",
        initialdir="/media/moybadajoz/Nuevo vol/DataSet With Vid",
    )
    print(dataset_path)
    if len(dataset_path) == 0 or dataset_path == "":
        print("No folder selected")
        return []
    subjects = glob.glob("*", root_dir=dataset_path)
    files = []
    for s in subjects:
        for i in range(4):
            mcd_file = f"{dataset_path}{split_char}{s}{split_char}mcd-{i}.csv"
            # aruco_file = f"{dataset_path}{split_char}{s}{split_char}arUcos-{i}.csv"
            if not Path(mcd_file).exists():
                continue
            files.append((mcd_file,))
    return files


def read_mcd_file(
    path: str,
    cutoff_freq=OPTIMAL_CONFIG["cutoff_freq"],
    filter_order=OPTIMAL_CONFIG["filter_order"],
) -> dict[str, np.ndarray]:
    """
    Read the mcd file and return a dictionary with the data.

    Args:
        path (str): Path to the MCD CSV file.
        cutoff_freq (float): Cutoff frequency for the low-pass filter in Hz (default: 9).
        filter_order (int): Order of the Butterworth filter (default: 6).
    """
    # abre el archivo y lo desglosa en diccionarios
    with open(path, "r") as file:
        measures = csv_comment(file.readline())
        mcd_file = [row for row in csv.DictReader(file, delimiter=";")]
    mcd_file = mcd_file[125:]
    # convierte los datos en un diccionario de numpy arrays tipo float
    print(len(mcd_file))
    mcd_data = {
        key: np.array(
            [np.array(data[key].split(","), dtype=float) for data in mcd_file]
        )
        for key in mcd_file[0].keys()
    }
    # 'aplana' el arrat de tiempo para que sea de una unica dimension
    mcd_data["time"] = mcd_data["time"].flatten()
    # agrega el diccionario de medidas
    mcd_data["measures"] = measures
    # obtiene la frecuencia de muestreo
    fs = 1000 / np.mean(np.diff(mcd_data["time"]))
    # fs=100
    # convierte los datos a unidades (rad/s y m/s^2) y aplica un filtro pasabajas
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
    # abre el archivo y lo desglosa en diccionarios
    with open(path, "r") as file:
        aruco_file = [row for row in csv.DictReader(file, delimiter=";")]
    # extrae las llaves
    aruco_keys = aruco_file[0].keys()
    # agrupa los datos de cada llave en una lista
    aruco_data_no_processed = {
        key: [data[key] for data in aruco_file] for key in aruco_keys
    }
    # extrae todas las marcas de tiempo en un array de enteros
    aruco_time = np.array([row["time"] for row in aruco_file], dtype=int)
    # liga cada dato con su respectiva marca de tiempo
    aruco_data = {
        key: {
            time: np.array(data.split(","), dtype=float)
            for time, data in zip(aruco_time, aruco_data_no_processed[key])
            if data != "" and time >= 0
        }
        for key in aruco_keys
        if key != "time"
    }
    # divide las marcas de tiempo y los datos en dos arrays, siguen ligadas por el indice
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
    Construye una matriz de transformación 4x4 a partir de una posición 3D y un cuaternión.

    Args:
        position (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa la posición.
        rotation: --
    Returns:
        numpy.ndarray: Matriz de transformación 4x4.
    """

    # 1. Matriz de Rotación
    rotation_matrix = rotation.as_matrix()

    # 2. Matriz de Traslación
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # 3. Matriz de Transformación Completa
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix


def apply_transformation(transformation_matrix, point):
    """
    Aplica una matriz de transformación a un punto.

    Args:
        transformation_matrix (numpy.ndarray): Matriz de transformación 4x4.
        point (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa el punto.

    Returns:
        numpy.ndarray: Arreglo numpy de 3 elementos [x', y', z'] que representa la nueva posición del punto.
    """

    # 1. Representación del punto en coordenadas homogéneas
    homogeneous_point = np.append(point, 1)  # add 1 to the end of the array
    # change the array to be a 4x1 array.
    homogeneous_point = homogeneous_point.reshape((4, 1))

    # 2. Multiplicación de matrices
    transformed_point = np.dot(transformation_matrix, homogeneous_point)

    # 3. Obtención de la nueva posición
    # take the first 3 elements, and flatten the array.
    new_point = transformed_point[:3].flatten()

    return new_point


def get_pose(quats, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pose = {}
    # matriz de transformacion para la cadera
    T_h = transformation_matrix([0, 0, 0], quats["hip"])
    # puntos de cadera
    pose["hip_r"] = apply_transformation(T_h, points["hip_r"])
    pose["hip_l"] = apply_transformation(T_h, points["hip_l"])
    # Matriz de tranformacion de las rodillas (respecto a la cadera)
    # formadas por la traslacion (posicion) de la cadera y la rotacion del femur
    T_kr = transformation_matrix(pose["hip_r"], quats["rf"])
    T_kl = transformation_matrix(pose["hip_l"], quats["lf"])
    # puntos de rodilla
    pose["knee_r"] = apply_transformation(T_kr, points["knee_r"])
    pose["knee_l"] = apply_transformation(T_kl, points["knee_l"])
    # Matriz de transformacion de los tobillos (respecto a las rodillas)
    # formadas por la traslacion (posicion) de la rodilla y la rotacion de la tibia
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


def angulo_gravedad(accel: np.ndarray) -> tuple[float, float]:
    """
    Calcula los ángulos de Roll y Pitch a partir del vector de aceleración
    asumiendo que el sensor está en estado estático.

    Args:
        accel (np.ndarray): Un arreglo de NumPy con las 3 componentes de la
                           aceleración [ax, ay, az].

    Returns:
        tuple[float, float]: Una tupla conteniendo (roll, pitch) en grados.
    """
    # Es una buena práctica normalizar el vector de aceleración primero.
    # Esto hace que el cálculo sea independiente de si la unidad es g o m/s^2.
    if np.linalg.norm(accel) == 0:
        return (0.0, 0.0)  # Evitar división por cero si el vector es nulo
    accel_norm = accel / np.linalg.norm(accel)

    ax, ay, az = accel_norm[0], accel_norm[1], accel_norm[2]

    # Calcular Roll (phi) en radianes
    roll_rad = np.arctan2(ay, az)

    # Calcular Pitch (theta) en radianes
    # La fórmula robusta usa la raíz cuadrada de ay^2 + az^2
    pitch_rad = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    # Convertir radianes a grados
    roll_deg = np.rad2deg(roll_rad)
    pitch_deg = np.rad2deg(pitch_rad)

    return (roll_deg, pitch_deg)


def reorient_data(
    accel: np.ndarray,
    gyro: np.ndarray,
    calibration_samples=OPTIMAL_CONFIG["calibration_samples"],
) -> tuple[np.ndarray, np.ndarray]:
    # obtiene la media de posicion en reposo
    rest = np.mean(accel[:calibration_samples], axis=0)
    rest /= np.linalg.norm(rest)
    # print(rest)
    # obtiene la matrix de rotacion para alinear el vector de gravedad con el eje z
    accel_R = rotation_matrix_from_vectors(rest)
    # reorienta los datos de aceleracion y giroscopio
    reoriented_accel = (accel_R @ accel.T).T
    reoriented_gyro = (accel_R @ gyro.T).T

    return reoriented_accel, reoriented_gyro


def get_orientation(
    mcd_data,
    pca_percentile=OPTIMAL_CONFIG["pca_percentile"],
    madgwick_gain=OPTIMAL_CONFIG["madgwick_gain"],
    slerp_clip_min=OPTIMAL_CONFIG["slerp_clip_min"],
    slerp_clip_max=OPTIMAL_CONFIG["slerp_clip_max"],
    calibration_samples=OPTIMAL_CONFIG["calibration_samples"],
    fusion_filter=OPTIMAL_CONFIG["fusion_filter"],
) -> dict[str, np.ndarray]:
    """
    Get the orientation of the sensors using Madgwick/EKF filter with PCA yaw correction and dynamic Slerp.
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
    # creacion de los filtros complementarios
    s_p = 1000 / np.mean(np.diff(mcd_data["time"]))

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

        # MÉTODO AVANZADO: PCA sobre el giroscopio para encontrar el eje de la bisagra de la rodilla
        gyro_data = reoriented_gyro[key]

        # Filtrar para usar solo los momentos de mayor movimiento (fase de oscilación de la marcha)
        magnitudes = np.linalg.norm(gyro_data, axis=1)
        threshold = np.percentile(
            magnitudes, pca_percentile
        )  # Usar solo el (100-pca_percentile)% superior de datos más activos
        movimiento = gyro_data[magnitudes > threshold]

        if len(movimiento) > 10:
            cov_matrix = np.cov(movimiento.T)
            cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

            if np.isfinite(cov_matrix).all():
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                principal_axis = np.real(eigenvectors[:, np.argmax(eigenvalues)])

                axis_xy = np.real(principal_axis[:2]).astype(float)
                norm_xy = np.linalg.norm(axis_xy)

                if np.isfinite(norm_xy) and norm_xy > 1e-6:
                    axis_xy = axis_xy / norm_xy

                    if axis_xy[1] < 0:
                        axis_xy = -axis_xy

                    yaw_angle = float(np.arctan2(axis_xy[1], axis_xy[0]))

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

    # Aplicar un modelo cinemático dinámico a la cadera (Slerp ponderado)
    hip_quats_list = []
    for i in range(len(mcd_data["time"])):
        vel_rf = np.linalg.norm(reoriented_gyro["rf"][i])
        vel_lf = np.linalg.norm(reoriented_gyro["lf"][i])
        total_vel = vel_rf + vel_lf

        if total_vel < 1e-5:
            weight = 0.5
        else:
            weight = vel_rf / total_vel

        weight = np.clip(weight, slerp_clip_min, slerp_clip_max)

        slerp = Slerp([0, 1], R.concatenate([quats["rf"][i], quats["lf"][i]]))
        hip_quats_list.append(slerp(weight).as_quat())

    quats["hip"] = R.from_quat(np.array(hip_quats_list))
    return quats


def mcd_reconstruction(
    mcd_data: dict[str, np.ndarray],
    pca_percentile=OPTIMAL_CONFIG["pca_percentile"],
    madgwick_gain=OPTIMAL_CONFIG["madgwick_gain"],
    slerp_clip_min=OPTIMAL_CONFIG["slerp_clip_min"],
    slerp_clip_max=OPTIMAL_CONFIG["slerp_clip_max"],
    calibration_samples=OPTIMAL_CONFIG["calibration_samples"],
    fusion_filter=OPTIMAL_CONFIG["fusion_filter"],
    pivot_height_threshold=OPTIMAL_CONFIG["pivot_height_threshold"],
    swing_velocity_threshold=OPTIMAL_CONFIG["swing_velocity_threshold"],
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
    pace = []
    for i in range(n):
        if i == 0:
            z_compare = 1 if direction == "R" else 0
        else:
            # 1. Cinemática (Diferencia de altura Z)
            diff_z = pose["ankle_l"][i][2] - pose["ankle_r"][i][2]

            # 2. Dinámica (Magnitud del giroscopio de las tibias)
            mag_gyro_rt = np.linalg.norm(mcd_data["gyro_RT"][i])
            mag_gyro_lt = np.linalg.norm(mcd_data["gyro_LT"][i])

            # Si la diferencia de altura supera el umbral, confiamos en la posición
            if abs(diff_z) > pivot_height_threshold:
                z_compare = diff_z
            else:
                z_compare = mag_gyro_lt - mag_gyro_rt

        v_rt = mcd_data["gyro_RT"][i, 1]
        v_lt = mcd_data["gyro_LT"][i, 1]

        # Filtro adicional de swing override
        if z_compare <= 0:
            if v_lt > swing_velocity_threshold and v_lt > v_rt:
                z_compare = 1
        else:
            if v_rt > swing_velocity_threshold and v_rt > v_lt:
                z_compare = -1

        # traslacion tobillo al origen
        if z_compare > 0:
            T_a_2_o = transformation_matrix(
                pose["ankle_r"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_ar
            pace.append(4)
        else:
            T_a_2_o = transformation_matrix(
                pose["ankle_l"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_al
            pace.append(0)

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
    # plt.figure()
    # plt.plot(pace, label="pace 0=L")
    # # plt.plot(pace2, label="pace2 0=L")
    # plt.plot(mcd_data["gyro_RT"][:, 1], label="gyro_RT")
    # plt.plot(mcd_data["gyro_LT"][:, 1], label="gyro_LT")
    # plt.legend()
    # plt.grid(1)
    # plt.show()

    # fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 4))
    # fig.suptitle(title)
    # # ax.plot(rot_rk[:, 0], label='Flexión-Extensión Z')
    # # ax.plot(rot_rk[:, 1], label='Flexión-Extensión X')
    # # ax.plot(rot_rk[:, 2], label='Flexión-Extensión Y')
    # ax[1, 1].set_title('Cadera Derecha')
    # ax[1, 1].plot(rot_r[:,0], label='rf Z')
    # ax[1, 1].plot(rot_r[:,1], label='rf x')
    # ax[1, 1].plot(rot_r[:,2], label='rf y')
    # ax[1, 1].legend()
    # ax[1, 1].grid(1)
    # ax[2, 1].set_title('Cadera Izquierda')
    # ax[2, 1].grid(1)
    # ax[2, 1].plot(rot_l[:,0], label='lf Z')
    # ax[2, 1].plot(rot_l[:,1], label='lf x')
    # ax[2, 1].plot(rot_l[:,2], label='lf y')
    # ax[2, 1].legend()
    # ax[1, 0].set_title('Rodilla Derecha')
    # ax[1, 0].grid(1)
    # ax[1, 0].plot(rot_rk[:,0], label='rk Z')
    # ax[1, 0].plot(rot_rk[:,1], label='rk x')
    # ax[1, 0].plot(rot_rk[:,2], label='rk y')
    # ax[1, 0].legend()
    # ax[2, 0].set_title('Rodilla Izquierda')
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


def mean_euclidian_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Euclidean error.
    """
    return np.linalg.norm(a - b, axis=1).mean()


def components_error(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return a - b


def root_mean_square_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the root mean square error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Root mean square error.
    """
    return np.sqrt(np.mean(np.linalg.norm(a - b, axis=1) ** 2))


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


def mean_absolute_error2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Mean absolute error.
    """
    return np.mean(np.abs(a - b), axis=1).mean()


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
    Calcula la velocidad de desplazamiento (m/s) a partir de posiciones 3D y marcas de tiempo.
    Recorta los períodos quietos al inicio y al final de la captura.

    Args:
        positions (np.ndarray): Array (N, 3) de posiciones 3D.
        timestamps_ms (np.ndarray): Array (N,) de marcas de tiempo en milisegundos.
        trim_ms (float): Milisegundos a recortar al inicio y al final para evitar
                         períodos estáticos que sesgarían el promedio.

    Returns:
        dict: Diccionario con:
            - 'velocities': array de velocidades instantáneas (m/s)
            - 'timestamps': array de timestamps correspondientes (ms)
            - 'mean_velocity': velocidad media (m/s)
            - 'std_velocity': desviación estándar de la velocidad
            - 'var_velocity': varianza de la velocidad
    """
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

    dp = np.diff(trimmed_pos, axis=0)
    dt = np.diff(trimmed_t) / 1000.0

    speeds = np.linalg.norm(dp, axis=1) / dt
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
    Calcula el ángulo de flexión de la rodilla a partir de las posiciones de
    cadera, rodilla y tobillo.

    Returns:
        float: Ángulo de flexión de la rodilla en grados (0° = extendida).
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
    """Calcula la serie temporal de ángulos de rodilla en grados."""
    n = min(len(hip_pos), len(knee_pos), len(ankle_pos))
    return np.array(
        [compute_knee_angle(hip_pos[i], knee_pos[i], ankle_pos[i]) for i in range(n)]
    )


def detect_stride_lengths(
    ankle_positions, timestamps_ms, min_distance_samples=50, prominence=0.005
):
    """
    Detecta longitudes de zancada a partir de heel strikes (mínimos en Z del tobillo).

    Args:
        ankle_positions (np.ndarray): Posiciones del tobillo (N, 3).
        timestamps_ms (np.ndarray): Marcas de tiempo en ms.
        min_distance_samples (int): Distancia mínima entre picos en muestras.
        prominence (float): Prominencia mínima para detectar un heel strike.

    Returns:
        tuple: (stride_lengths_m, stride_times_ms)
    """
    z = ankle_positions[:, 2]
    neg_z = -z
    peaks, _ = find_peaks(neg_z, distance=min_distance_samples, prominence=prominence)

    if len(peaks) < 2:
        return np.array([]), np.array([])

    stride_lengths = []
    stride_times = []
    for i in range(len(peaks) - 2):
        p1 = ankle_positions[peaks[i]]
        p2 = ankle_positions[peaks[i + 2]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        stride_len = np.sqrt(dx**2 + dy**2)
        if 0.1 < stride_len < 3.0:
            stride_lengths.append(stride_len)
            stride_times.append(timestamps_ms[peaks[i]])

    return np.array(stride_lengths), np.array(stride_times)


def process_files(
    files: tuple[str, ...],
    use_3d=False,
    plot=True,
    return_tuple=True,
    cutoff_freq=OPTIMAL_CONFIG["cutoff_freq"],
    filter_order=OPTIMAL_CONFIG["filter_order"],
    pca_percentile=OPTIMAL_CONFIG["pca_percentile"],
    madgwick_gain=OPTIMAL_CONFIG["madgwick_gain"],
    slerp_clip_min=OPTIMAL_CONFIG["slerp_clip_min"],
    slerp_clip_max=OPTIMAL_CONFIG["slerp_clip_max"],
    calibration_samples=OPTIMAL_CONFIG["calibration_samples"],
    fusion_filter=OPTIMAL_CONFIG["fusion_filter"],
    pivot_height_threshold=OPTIMAL_CONFIG["pivot_height_threshold"],
    swing_velocity_threshold=OPTIMAL_CONFIG["swing_velocity_threshold"],
) -> any:
    # Lee los datos de los archivos MCD (sensores inerciales) y ArUco (ground truth).
    mcd_data = read_mcd_file(
        files[0], cutoff_freq=cutoff_freq, filter_order=filter_order
    )

    # Determina la dirección de la marcha y la pierna que no se ve (oculta).
    direction = mcd_data["measures"]["direction"]
    hidden_leg = "L" if direction == "R" else "R"

    has_aruco = len(files) > 1 and files[1] is not None and Path(files[1]).exists()

    if has_aruco:
        aruco_data = read_aruco_file(files[1])
        aruco_data = adjust_aruco_data(aruco_data, direction)
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

        femur_length, tibia_length = measures_calculation(aruco_processed, direction)
        mcd_data["measures"]["femur_length"] = femur_length
        mcd_data["measures"]["tibia_length"] = tibia_length
    else:
        if (
            "femur_length" not in mcd_data["measures"]
            or mcd_data["measures"]["femur_length"] <= 0.1
        ):
            mcd_data["measures"]["femur_length"] = 0.42
        if (
            "tibia_length" not in mcd_data["measures"]
            or mcd_data["measures"]["tibia_length"] <= 0.1
        ):
            mcd_data["measures"]["tibia_length"] = 0.45

    if (
        "hip_width" not in mcd_data["measures"]
        or mcd_data["measures"]["hip_width"] <= 0.05
    ):
        mcd_data["measures"]["hip_width"] = 0.25

    # Realiza la reconstrucción del esqueleto a partir de los datos de los sensores inerciales.
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

    # Define un ángulo de rotación objetivo para alinear la reconstrucción con el sistema de coordenadas global.
    z_target = 0 if direction == "L" else np.pi
    # Crea una matriz de transformación para aplicar la corrección de rotación.
    T_fix = transformation_matrix([0, 0, 0], R.from_euler("ZXY", [z_target, 0, 0]))
    # Aplica la matriz de transformación a todos los puntos de la reconstrucción.
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

    # Si no se proporciona archivo ArUco (modo test contra curvas o sin GT)
    if not has_aruco:
        trajectory = np.mean(np.array(list(reconstruction.values())), axis=0)
        shape = "lenmiscata"
        if shape == "lenmiscata":
            a = 1.75
            t = np.linspace(0, 2 * np.pi, 1000)
            denominador = 1 + np.sin(t) ** 2
            numerador_const = a * np.sqrt(2)
            x = (numerador_const * np.cos(t)) / denominador
            y = (numerador_const * np.sin(t) * np.cos(t)) / denominador
            if files[0].split(".")[0][-1] == "5":
                x_r = x * np.cos(-np.pi / 4) - y * np.sin(-np.pi / 4)
                y_r = x * np.sin(-np.pi / 4) + y * np.cos(-np.pi / 4)
            else:
                x_r = x * np.cos(np.pi / 4) - y * np.sin(np.pi / 4)
                y_r = x * np.sin(np.pi / 4) + y * np.cos(np.pi / 4)
        elif shape == "hourglass":
            x = np.array([-2, 2, 0, 0, -2])
            y = np.array([0, 0, 2, -2, 0])
            x_r = np.empty(0)
            y_r = np.empty(0)
            for i in range(1, len(x)):
                x_r = np.append(x_r, np.linspace(x[i - 1], x[i], 250))
                y_r = np.append(y_r, np.linspace(y[i - 1], y[i], 250))
        forma = np.array([x_r, y_r]).T
        errors = np.array(
            [distancia_punto_forma2(pt, forma) for pt in trajectory[:, :2]]
        )
        print(f"error medio ({files[0].split(split_char)[-1]}): {errors.mean()}")

        plt.figure()
        plt.plot(forma[:, 0], forma[:, 1], label="Ground Truth")
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Reconstruction")
        plt.legend()
        plt.grid(True)  # Añadir una cuadrícula
        plt.yticks(np.arange(-2.5, 2.5, 0.5))
        plt.xticks(np.arange(-2.5, 2.5, 0.5))
        # plt.title(f"lenmiscata-{files[0].split(split_char)[-1].split('.')[0][-1]}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")

        # plt.figure()
        # plt.plot(errors)
        # plt.grid(True)  # Añadir una cuadrícula
        # plt.title(f"lenmiscata-{files[0].split(split_char)[-1].split('.')[0][-1]}")

        capture_id = files[0].split(split_char)[-1].split(".")[0][-1]
        os.makedirs("Results/lenmiscata", exist_ok=True)
        os.makedirs(f"Results/{shape}", exist_ok=True)
        try:
            plt.savefig(
                f"Results/lenmiscata/lenmiscata-{capture_id}.tiff",
                dpi=300,
                bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except Exception:
            pass
        plt.savefig(
            f"Results/{shape}-{capture_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            f"Results/lenmiscata/lenmiscata-{capture_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

        return errors.mean()

    # Homogeniza los datos, asegurando que los datos de MCD y ArUco tengan las mismas marcas de tiempo.
    homogeneous_data = {}
    for key in reconstruction.keys():
        if key == f"hip_{hidden_leg.lower()}":
            continue
        t, a_out, b_out = homogenize_3dpoint_data(
            reconstruction[key],
            aruco_processed[key]["data"],
            mcd_data["time"],
            aruco_processed[key]["time"],
        )
        trim_min = t > 1000
        trim_max = t < t.max() - 500
        trim = trim_min & trim_max
        homogeneous_data[key] = {
            "time": t[trim],
            "mcd": a_out[trim],
            "aruco": b_out[trim],
        }

    common_time = set()
    for key in homogeneous_data.keys():
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
        homogeneous_data[key]["eu_var"] = np.var(homogeneous_data[key]["euclidean"])
        homogeneous_data[key]["eu_std"] = np.std(homogeneous_data[key]["euclidean"])
        if len(common_time) == 0:
            common_time = set(homogeneous_data[key]["time"])
        else:
            common_time = common_time.intersection(set(homogeneous_data[key]["time"]))

    common_time = sorted(list(common_time))
    mcd_common = np.empty((0, len(common_time), 3))
    aruco_common = np.empty((0, len(common_time), 3))
    for key in homogeneous_data.keys():
        isin = np.isin(homogeneous_data[key]["time"], common_time)
        mcd_common = np.append(mcd_common, [homogeneous_data[key]["mcd"][isin]], axis=0)
        aruco_common = np.append(
            aruco_common, [homogeneous_data[key]["aruco"][isin]], axis=0
        )

    gen_eucliden_error = np.empty((0))
    for i in range(len(common_time)):
        d_position = np.mean(mcd_common[:, i, :], axis=0)
        m_position = np.mean(aruco_common[:, i, :], axis=0)
        err = euclidean_error_3Dpoint(d_position, m_position)
        gen_eucliden_error = np.append(gen_eucliden_error, err)

    if use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
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

    if plot:
        fig_2d = plt.figure(figsize=figsize)
        for key in ["hip_r", "ankle_r"]:
            if key in homogeneous_data:
                plt.plot(
                    homogeneous_data[key]["mcd"][:, 0],
                    homogeneous_data[key]["mcd"][:, 1],
                    label=f"MCD {key}",
                )
                plt.plot(
                    homogeneous_data[key]["aruco"][:, 0],
                    homogeneous_data[key]["aruco"][:, 1],
                    label=f"ArUco {key}",
                    linestyle="--",
                )
        file_stem = Path(files[0]).stem
        plt.title(f"Trayectoria 2D - {file_stem}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        os.makedirs("Results/trajectory", exist_ok=True)
        plt.savefig(
            f"Results/trajectory/{file_stem}_2d.png",
            dpi=300,
            bbox_inches="tight",
        )

    # --- Cálculo de velocidad de desplazamiento ---
    velocity_data = {}
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
    # MÉTRICAS CIENTÍFICAS ADICIONALES
    # =====================================================================
    extra_metrics = {}

    # --- 1. RMSE por articulación ---
    rmse_per_joint = {}
    for key in homogeneous_data.keys():
        diff = homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"]
        rmse = np.sqrt(np.mean(diff**2))
        rmse_per_joint[key] = rmse
    extra_metrics["rmse_per_joint"] = rmse_per_joint
    extra_metrics["rmse_mean"] = np.mean(list(rmse_per_joint.values()))

    # --- 2. Correlación de Pearson por eje ---
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

    # --- 3. Datos para Bland-Altman (promedio general de posición) ---
    bland_altman = {}
    for ax_i, ax_name in enumerate(axis_labels):
        all_mcd_ax = np.concatenate(
            [homogeneous_data[key]["mcd"][:, ax_i] for key in homogeneous_data.keys()]
        )
        all_aruco_ax = np.concatenate(
            [homogeneous_data[key]["aruco"][:, ax_i] for key in homogeneous_data.keys()]
        )
        diff_vals = all_mcd_ax - all_aruco_ax
        bland_altman[ax_name] = {
            "mean": (all_mcd_ax + all_aruco_ax) / 2.0,
            "diff": diff_vals,
            "bias": np.mean(diff_vals),
            "std_diff": np.std(diff_vals),
            "upper_loa": np.mean(diff_vals) + 1.96 * np.std(diff_vals),
            "lower_loa": np.mean(diff_vals) - 1.96 * np.std(diff_vals),
        }
    extra_metrics["bland_altman"] = bland_altman

    # --- 4. Longitud de zancada ---
    stride_data = {}
    for side in ["r", "l"]:
        ankle_key = f"ankle_{side}"
        rec_strides, rec_stride_t = detect_stride_lengths(
            reconstruction[ankle_key], mcd_data["time"]
        )
        stride_data[f"reconstruction_{ankle_key}"] = {
            "lengths": rec_strides,
            "times": rec_stride_t,
            "mean": np.mean(rec_strides) if len(rec_strides) > 0 else 0.0,
            "std": np.std(rec_strides) if len(rec_strides) > 0 else 0.0,
        }
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

    # --- 5. ROM de rodilla (Range of Motion) ---
    knee_rom = {}
    for side in ["r", "l"]:
        hip_key = f"hip_{side}"
        knee_key = f"knee_{side}"
        ankle_key = f"ankle_{side}"

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

    if not return_tuple:
        return float(np.mean(gen_eucliden_error))

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
            break
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    files = get_dataset2()
    if len(files) == 0:
        print("No files selected")
        exit(1)
    # print(f'capture | x_r | x_p-value | y_r | y_p-value | z_r | z_p-value')
    gen_errors = [process_files(f) for f in files]
    if len(gen_errors) > 0 and isinstance(gen_errors[0], (tuple, list)):
        mean_errors = [np.mean(r[0]) for r in gen_errors]
        print(f"error medio general: {np.mean(mean_errors)}")
    else:
        print(f"error medio general: {np.mean(gen_errors)}")
    plt.show()
    #
