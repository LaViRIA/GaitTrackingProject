# Motion Capture Device

---

![Interfaz Capture.py](./Docs/img/device.jpg)

### Resumen

Este es un proyecto para capturar el movimiento del tren inferior de una persona, haciendo uso de sensores inerciales.

### Funcionamiento

Este proyecto se compone de cuatro secciones: cliente, host, api, aplicacion.

- **Cliente:** Son los dispositivos que se colocan en la persona, estos se componen en escencia de un ESP32C6 y un sensor inercial MPU6050, ademas de otros aditamentos para su uso como una bateria, un boton y un led RGB.
  El esp32 lee al sensor y envia los datos al host.
  [Documentacion del cliente](./Docs/es/clients.md)
- **Host:** Es un ESP32 conectado por Serial a un ordenador y por ESPNOW a los clientes, es un intermediado entre los clientes y el ordenador y la API.
  Se encarga de enviar los mensajes desde la API a los clientes ademas de gestionar su conexion con ellos y empaquetar los datos de los clientes para enviarlos a la API.
  [Documentacion del Host](./Docs/es/host.md)
- **API:** Script de python para gestionar la comunicacion entre la aplicacion y el host.
  Se encarga de permitir a la aplicacion de interactuar con el dispositivo en general.
  [Documentacion de la API](./Docs/es/api.md)
- **Aplicacion:** Aplicacion final para el usuario, puede haber de muchos tipos y funciones.
  Se encarga de darle uso al dispositivo.
  [Documentacion de la aplicacion (capture.py)](./Docs/es/capture.md)

### Capture

La aplicacion de captura _[capture.py](./capture.py)_ consta de todas las herramientas y funciones para realizar una captura de movimiento, con excepcion de una camara, la cual no es estrictamente necesaria para realizar una captura, aunque si para el _[posprocesado](./Docs/es/posprocess_manager.md)_

### [Demo](./demo.py)

Un software con la unica finalidad de hacer una demostracion en tiempo real de los datos obtenidos por los sensores.

### Posprocesado

Un script destinado a procesar imagenes de una grabacion, acompañado de un archivo csv con la captura de movimiento correspondiente, este script de encarga de obtener la posicion espacial de los arucos asociados a las articulaciones principales del tren inferior para cada frame y posteriormente sincronizarlas con los datos de la captura de movimiento dando como resultado otro archivo csv. [Mas informacion](./Docs/es/posprocess.md)

### Posprocess Manager

Un script que gestiona el posprocesado de diversas capturas en paralelo, permite agilizar el posprocesado. [Mas informacion](./Docs/es/posprocess_manager.md)

### Process All

Ees un script de análisis y validación que procesa masivamente conjuntos de datos (MCD y ArUco). Se encarga de reconstruir la cinemática del tren inferior a partir de los datos de los sensores inerciales y compara los resultados con la referencia óptica (ArUco) para calcular errores de posición, generar gráficas de rendimiento y evaluar la precisión del sistema.
