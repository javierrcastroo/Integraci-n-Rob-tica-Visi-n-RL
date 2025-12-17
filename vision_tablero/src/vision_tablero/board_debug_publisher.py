#!/usr/bin/env python3
"""Publicador de layouts de tablero en modo debug (sin cámaras).

Permite inicializar un tablero con barcos y munición fijos y actualizarlos
vía comandos por tópico. Responde a ``battleship/board_request`` igual que el
nodo real de visión.
"""

import json
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rospy
from std_msgs.msg import String

Cell = Tuple[int, int]


class BoardDebugPublisher:
    def __init__(self) -> None:
        rospy.init_node("board_debug_publisher", anonymous=True)

        self.board_name = rospy.get_param("~board_name", "debug_board")
        self.board_size = int(rospy.get_param("~board_size", 5))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate", 1.0))
        self.auto_publish = bool(rospy.get_param("~auto_publish", True))

        self.ship_two_cells: Set[Cell] = set()
        self.ship_one_cells: Set[Cell] = set()
        self.ammo_cells: Set[Cell] = set()

        self._load_initial_cells("ship_two_cells", self.ship_two_cells)
        self._load_initial_cells("ship_one_cells", self.ship_one_cells)
        self._load_initial_cells("ammo_cells", self.ammo_cells)

        self.board_pub = rospy.Publisher("battleship/board_layout", String, queue_size=10)
        self.request_sub = rospy.Subscriber(
            "battleship/board_request", String, self._request_cb, queue_size=10
        )
        self.cmd_sub = rospy.Subscriber(
            "battleship/debug_layout_cmd", String, self._layout_cmd_cb, queue_size=10
        )

        rospy.loginfo(
            "[board_debug_publisher] Tablero '%s' iniciado (%dx%d). Barcos2=%s, Barcos1=%s, Municion=%s",
            self.board_name,
            self.board_size,
            self.board_size,
            sorted(self.ship_two_cells),
            sorted(self.ship_one_cells),
            sorted(self.ammo_cells),
        )

        # Lanzamos un hilo opcional para repintar a intervalos.
        if self.auto_publish and self.publish_rate_hz > 0:
            self._timer_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._timer_thread.start()

        # Publicación inicial para que el resto de nodos tengan datos sin esperar petición.
        self.publish_layout(reason="startup")

    # ----------------------- helpers de inicialización -----------------------
    def _load_initial_cells(self, param: str, target: Set[Cell]) -> None:
        raw = rospy.get_param(f"~{param}", "")
        if isinstance(raw, str):
            items = [t for t in raw.replace(",", " ").split() if t]
        elif isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = []

        for item in items:
            cell = self._parse_cell(item)
            if cell is not None:
                target.add(cell)

    # ----------------------------- callbacks -----------------------------
    def _request_cb(self, msg: String) -> None:
        self.publish_layout(reason=msg.data or "request")

    def _layout_cmd_cb(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        tokens = text.split()
        verb = tokens[0].lower()

        if verb in {"add", "set", "ship2", "ship_two"} and len(tokens) >= 2:
            self._handle_add(tokens[1:], target=self.ship_two_cells, label="ship_two")
        elif verb in {"ship1", "ship_one"} and len(tokens) >= 2:
            self._handle_add(tokens[1:], target=self.ship_one_cells, label="ship_one")
        elif verb in {"ammo", "municion"} and len(tokens) >= 2:
            self._handle_add(tokens[1:], target=self.ammo_cells, label="ammo")
        elif verb in {"clear", "del", "rm"} and len(tokens) >= 2:
            self._handle_clear(tokens[1:])
        elif verb == "reset":
            self.ship_two_cells.clear()
            self.ship_one_cells.clear()
            self.ammo_cells.clear()
            rospy.loginfo("[board_debug_publisher] Layout reseteado")
        elif verb == "publish":
            self.publish_layout(reason="manual")
        elif verb == "list":
            rospy.loginfo(
                "[board_debug_publisher] Estado actual -> ship2=%s ship1=%s ammo=%s",
                sorted(self.ship_two_cells),
                sorted(self.ship_one_cells),
                sorted(self.ammo_cells),
            )
        else:
            rospy.logwarn(
                "[board_debug_publisher] Comando no reconocido: '%s' (usa add/ship2/ship1/ammo/clear/reset/publish/list)",
                text,
            )

    # ----------------------------- lógica -----------------------------
    def _handle_add(self, cells: Sequence[str], target: Set[Cell], label: str) -> None:
        added: List[Cell] = []
        for raw in cells:
            cell = self._parse_cell(raw)
            if cell is None:
                rospy.logwarn("[board_debug_publisher] Celda inválida: %s", raw)
                continue
            target.add(cell)
            added.append(cell)
        if added:
            rospy.loginfo("[board_debug_publisher] Añadidas %s en %s", added, label)

    def _handle_clear(self, cells: Sequence[str]) -> None:
        cleared: List[Cell] = []
        for raw in cells:
            cell = self._parse_cell(raw)
            if cell is None:
                continue
            for bucket in (self.ship_two_cells, self.ship_one_cells, self.ammo_cells):
                if cell in bucket:
                    bucket.discard(cell)
                    cleared.append(cell)
        if cleared:
            rospy.loginfo("[board_debug_publisher] Celdas eliminadas: %s", cleared)

    def _publish_loop(self) -> None:
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            self.publish_layout(reason="auto")
            rate.sleep()

    def publish_layout(self, *, reason: str) -> None:
        board = self._build_board_dict()
        payload = {"boards": [board]}
        msg = String()
        msg.data = json.dumps(payload)
        self.board_pub.publish(msg)
        log_fn = rospy.logdebug if reason == "auto" else rospy.loginfo
        log_fn(
            "[board_debug_publisher] Layout publicado (%s): ship2=%s ship1=%s ammo=%s",
            reason,
            sorted(self.ship_two_cells),
            sorted(self.ship_one_cells),
            sorted(self.ammo_cells),
        )

    def _build_board_dict(self) -> Dict:
        board = {
            "name": self.board_name,
            "board_size": self.board_size,
            "ship_two_cells": self._cells_to_list(self.ship_two_cells),
            "ship_one_cells": self._cells_to_list(self.ship_one_cells),
            "ammo_cells": self._cells_to_list(self.ammo_cells),
            "ship_two_positions": [],
            "ship_one_positions": [],
            "ammo_positions": [],
            "ship_two_detections": [],
            "ship_one_detections": [],
            "ammo_detections": [],
        }
        board["cells"] = self._cells_with_type(
            board["ship_two_cells"], board["ship_one_cells"], board["ammo_cells"]
        )
        return board

    @staticmethod
    def _cells_with_type(ship_two_cells: List[List[int]], ship_one_cells: List[List[int]], ammo_cells: List[List[int]]) -> List[Dict]:
        cells: List[Dict] = []
        for r, c in ship_two_cells:
            cells.append({"row": r, "col": c, "type": "ship_two"})
        for r, c in ship_one_cells:
            cells.append({"row": r, "col": c, "type": "ship_one"})
        for r, c in ammo_cells:
            cells.append({"row": r, "col": c, "type": "ammo"})
        return cells

    @staticmethod
    def _cells_to_list(cells: Iterable[Cell]) -> List[List[int]]:
        return [[r, c] for r, c in sorted(cells)]

    @staticmethod
    def _parse_cell(text: str) -> Optional[Cell]:
        if not text:
            return None
        t = text.strip().upper()
        if len(t) < 2:
            return None
        letter = t[0]
        number = t[1:]
        if not letter.isalpha() or not number.isdigit():
            return None
        row = ord(letter) - ord("A")
        col = int(number) - 1
        if row < 0 or col < 0:
            return None
        return row, col


def main() -> None:
    _node = BoardDebugPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
