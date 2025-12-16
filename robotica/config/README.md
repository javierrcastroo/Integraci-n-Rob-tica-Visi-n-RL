# Configuración de la pose del Aruco

Este paquete utiliza el archivo `poseAruco.yaml` para definir la pose del marcador Aruco fijo respecto a la base del robot. Las posiciones y orientaciones capturadas en ese YAML son las coordenadas en las que la punta del robot coincide físicamente con el marcador fijo.

## Cómo usar la pose para convertir coordenadas

1. **Definir frames**: considera `\{R\}` como el frame de la base del robot y `\{A\}` como el frame del Aruco fijo. El YAML almacena la transformación `^R T_A`, es decir, la posición y orientación de `\{A\}` respecto a `\{R\}`.
2. **Construir la transformación**: a partir de `poseAruco.yaml`, forma la matriz homogénea `^R T_A` con el vector de traslación `(x, y, z)` y el cuaternión `(x, y, z, w)` para la rotación.
3. **Transformar objetivos detectados**: si la visión devuelve un punto `p_A` (coordenadas de un barco) expresado en `\{A\}`, obtén su posición en el robot como:
   \[
   p_R =
   ^R T_A \cdot
   \begin{bmatrix}
   p_A \\
   1
   \end{bmatrix}
   \]
   donde `p_R` son las coordenadas del objetivo respecto al robot.
4. **Aplicar en código**: en ROS puedes usar `tf`/`tf2` o multiplicación de matrices/cuaterniones para generar `^R T_A` una sola vez al arrancar, y luego convertir cada detección de Aruco a `p_R` antes de enviar el objetivo al controlador del robot.

Con esta relación fija entre `\{R\}` y `\{A\}`, cualquier coordenada medida respecto al marcador Aruco se puede traducir de forma determinista al frame del robot.

## Dónde se usa en el código

El nodo `robotica/src/robotica/robot_attack_node.py` ya carga `Pose_Actual` (proveniente de `poseAruco.yaml`) al arrancar. A partir de esa pose calcula:

- `aruco_origin_x/aruco_origin_y`: traslación del centro del ArUco al frame `base_link`.
- `aruco_yaw`: rotación alrededor de Z que alinea el tablero con el robot.

Luego, cada vez que recibe una celda del tablero, hace la conversión tablero→robot en `_board_to_base_xy`, usando esa traslación y rotación fija, antes de enviar la orden de movimiento al controlador del robot.
