# Motion Capture Device

---

![Capture.py Interface](./Docs/img/device.jpg)

### Overview

This is a project to capture the movement of a person's lower body, using inertial sensors.

### Operation

This project consists of four sections: client, host, api, application.

- **Client:** These are the devices placed on the person; they essentially consist of an ESP32C6 and an MPU6050 inertial sensor, along with other accessories for use such as a battery, a button, and an RGB LED.
  The ESP32 reads the sensor and sends the data to the host.
  [Client Documentation](./Docs/en/clients.md)
- **Host:** It is an ESP32 connected via Serial to a computer and via ESPNOW to the clients; it acts as an intermediary between the clients, the computer, and the API.
  It is responsible for sending messages from the API to the clients, managing their connection, and packaging client data to send to the API.
  [Host Documentation](./Docs/en/host.md)
- **API:** Python script to manage communication between the application and the host.
  It allows the application to interact with the device in general.
  [API Documentation](./Docs/en/api.md)
- **Application:** Final application for the user; there can be many types and functions.
  It is responsible for making use of the device.
  [Application Documentation (capture.py)](./Docs/en/capture.md)

### Capture

The capture application _[capture.py](./capture.py)_ consists of all the tools and functions to perform motion capture, with the exception of a camera, which is not strictly necessary to perform a capture, although it is for _[post-processing](./Docs/en/posprocess_manager.md)_

### [Demo](./demo.py)

Software with the sole purpose of demonstrating in real-time the data obtained by the sensors.

### Post-processing

A script designed to process images from a recording, accompanied by a csv file with the corresponding motion capture. This script is responsible for obtaining the spatial position of the ArUco markers associated with the main joints of the lower body for each frame and subsequently synchronizing them with the motion capture data, resulting in another csv file. [More information](./Docs/en/posprocess.md)

### Post-process Manager

A script that manages the post-processing of multiple captures in parallel, allowing for faster post-processing. [More information](./Docs/en/posprocess_manager.md)

### Process All

It is an analysis and validation script that massively processes datasets (MCD and ArUco). It is responsible for reconstructing the lower body kinematics from the inertial sensor data and comparing the results with the optical reference (ArUco) to calculate position errors, generate performance graphs, and evaluate system accuracy.
