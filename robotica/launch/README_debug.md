# Modo debug sin cámaras

Este lanzamiento arranca un tablero simulado y un ejecutor que acepta destinos manuales.

## Lanzar
```
roslaunch robotica debug_board_robot.launch start_moveit:=false start_rviz:=false
```
Carga `poseAruco.yaml` para fijar la pose del marcador como en el modo normal. Si tienes el paquete `ur3e_203_moveit_config` puedes activar `start_moveit:=true` y `start_rviz:=true` para ver la escena en RViz.

> Usa una segunda terminal (con el mismo entorno `source`-ado) para publicar los comandos de layout y de destino con `rostopic pub`.

## Cambiar el layout simulado
- El nodo `board_debug_publisher` publica `battleship/board_layout` continuamente.
- Ajusta los barcos/munición publicando comandos en `battleship/debug_layout_cmd` (ejemplos):
  - `add A1 B2` (añade barcos de 2)
  - `ship1 C3` (añade barco de 1)
  - `ammo D4` (añade munición)
  - `clear A1` (elimina la celda de cualquier lista)
  - `list`, `reset`, `publish`

## Enviar un objetivo al robot
Publica la celda en `battleship/debug_target` (formato A1, B3...). El ejecutor `robot_attack_debug` utiliza el layout simulado para cargar obstáculos y planifica un movimiento lineal hasta la celda indicada.

- La triangulación tablero→robot es idéntica al nodo normal: usa `poseAruco.yaml` (`Pose_Actual`) y los offsets/yaw configurados. Cada vez que publicas un objetivo, verás en consola los mismos logs `[robot_attack_executor]` del modo normal (robot→aruco, aruco→ficha, robot→ficha y pose final).
