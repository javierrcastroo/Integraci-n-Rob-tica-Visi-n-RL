"""Herramientas compartidas para el reconocimiento y la máquina de estados de gestos."""
from __future__ import annotations

from collections import Counter, deque
"""Componentes compartidos para el reconocimiento de gestos."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


def majority_vote(labels: Iterable[Optional[str]]) -> Optional[str]:
    """Devuelve la etiqueta más frecuente (o ``None`` si no hay votos)."""
    filtered = [lbl for lbl in labels if lbl is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


def majority_label_with_exclusions(
    labels: Sequence[Optional[str]], extra_invalid: Iterable[str] = ()
) -> tuple[Optional[str], int]:
    """
    Selecciona la etiqueta con mayor frecuencia ignorando valores no válidos.

    Parameters
    ----------
    labels:
        Ventana de etiquetas a evaluar.
    extra_invalid:
        Etiquetas adicionales que deben descartarse además de ``None`` y
        ``"????"``.
    """
    invalid = {None, "????"}
    invalid.update(extra_invalid)
    filtered = [lbl for lbl in labels if lbl not in invalid]
    if not filtered:
        return None, 0
    counts = Counter(filtered)
    label, count = counts.most_common(1)[0]
    return label, count


class GestureConsensus:
    """Gestor de consenso temporal sobre las etiquetas detectadas."""

    def __init__(self, smoothing: int = 7, stability_window: int = 150) -> None:
        self._recent = deque(maxlen=smoothing)
        self._window = deque(maxlen=stability_window)

    def step(
        self, label: Optional[str], extra_invalid: Iterable[str] = ()
    ) -> tuple[Optional[str], Optional[str], int]:
        """
        Introduce una nueva etiqueta y devuelve (label_estable, consenso, cuenta).
        """
        if label is not None:
            self._recent.append(label)
        stable_label = majority_vote(self._recent)
        self._window.append(stable_label)

        consensus_label: Optional[str] = None
        count = 0
        if len(self._window) == self._window.maxlen:
            consensus_label, count = majority_label_with_exclusions(
                list(self._window), extra_invalid
            )
            self._window.clear()
        return stable_label, consensus_label, count

    def clear(self) -> None:
        self._recent.clear()
        self._window.clear()


@dataclass
class SequenceEvent:
    """Evento generado por la máquina de estados de gestos."""

    kind: str
    message: str
    level: str = "info"
    label: Optional[str] = None
    count: int = 0
    actions: Optional[List[str]] = None
    reset_consensus: bool = False


class GestureSequenceManager:
    """Controla el armado, confirmación y guardado de gestos."""

    def __init__(
        self,
        arm_label: str = "demonio",
        save_label: str = "cool",
        confirm_label: str = "ok",
        reject_label: str = "nook",
        max_actions: int = 2,
    ) -> None:
        self.arm_label = arm_label
        self.save_label = save_label
        self.confirm_label = confirm_label
        self.reject_label = reject_label
        self.max_actions = max_actions

        self._armed = False
        self._actions: List[str] = []
        self.pending_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Propiedades de estado
    # ------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> List[str]:
        return list(self._actions)

    # ------------------------------------------------------------------
    # Gestión de máquina de estados automática
    # ------------------------------------------------------------------
    def handle_candidate(self, label: Optional[str], count: int) -> List[SequenceEvent]:
        events: List[SequenceEvent] = []
        if label is None:
            return events

        if label == self.arm_label:
            if not self._armed:
                self._armed = True
                self._actions.clear()
                self.pending_label = None
                events.append(
                    SequenceEvent(
                        kind="armed",
                        level="info",
                        message=f"[INFO] Secuencia armada tras gesto '{self.arm_label}'.",
                    )
                )
            return events

        if label == self.save_label:
            if self._armed:
                if self._actions:
                    events.append(
                        SequenceEvent(
                            kind="save",
                            level="info",
                            message=f"[INFO] Secuencia guardada tras gesto '{self.save_label}': {self._actions}",
                            actions=self.actions,
                        )
                    )
                else:
                    events.append(
                        SequenceEvent(
                            kind="warn",
                            level="warn",
                            message=f"[WARN] Gesto '{self.save_label}' recibido pero la lista está vacía.",
                        )
                    )
                self.reset()
                events.append(
                    SequenceEvent(
                        kind="reset",
                        level="info",
                        message="[INFO] Secuencia reiniciada, realiza 'demonio' para armar de nuevo.",
                        reset_consensus=True,
                    )
                )
            else:
                events.append(
                    SequenceEvent(
                        kind="warn",
                        level="warn",
                        message=f"[WARN] Ignorando '{self.save_label}' sin haber armado la secuencia.",
                    )
                )
            return events

        if label == self.confirm_label:
            if not self._armed:
                events.append(
                    SequenceEvent(
                        kind="warn",
                        level="warn",
                        message=f"[WARN] Ignorando '{self.confirm_label}' sin haber armado la secuencia con '{self.arm_label}'.",
                    )
                )
            elif self.pending_label is None:
                events.append(
                    SequenceEvent(
                        kind="warn",
                        level="warn",
                        message="[WARN] No hay coordenada pendiente que confirmar.",
                    )
                )
            elif self.action_count >= self.max_actions:
                events.append(
                    SequenceEvent(
                        kind="warn",
                        level="warn",
                        message="[WARN] Lista llena. Usa gesto 'cool' para guardar y reiniciar.",
                    )
                )
            else:
                label_added = self._append_action(self.pending_label)
                events.append(
                    SequenceEvent(
                        kind="confirm",
                        level="info",
                        message=(
                            f"[INFO] Coordenada '{label_added}' confirmada tras gesto '{self.confirm_label}' (cuenta={count})."
                        ),
                        label=label_added,
                        count=count,
                    )
                )
                self.pending_label = None
            return events

        if label == self.reject_label:
            if self.pending_label is not None:
                events.append(
                    SequenceEvent(
                        kind="reject",
                        level="info",
                        message=(
                            f"[INFO] Coordenada '{self.pending_label}' descartada tras gesto '{self.reject_label}'."
                        ),
                        label=self.pending_label,
                    )
                )
                self.pending_label = None
            else:
                events.append(
                    SequenceEvent(
                        kind="warn",
                        level="warn",
                        message="[WARN] 'nook' recibido pero no hay coordenada pendiente.",
                    )
                )
            return events

        # Otros gestos candidatos
        if not self._armed:
            events.append(
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message=(
                        f"[WARN] Ignorando gesto '{label}' sin armar la secuencia con '{self.arm_label}'."
                    ),
                )
            )
            return events

        if self.action_count >= self.max_actions:
            events.append(
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message="[WARN] Lista llena. Usa gesto 'cool' para guardar y reiniciar.",
                )
            )
            return events

        if self.pending_label == label:
            events.append(
                SequenceEvent(
                    kind="info",
                    level="info",
                    message=f"[INFO] Continúa pendiente la coordenada '{label}'.",
                    label=label,
                    count=count,
                )
            )
        else:
            self.pending_label = label
            events.append(
                SequenceEvent(
                    kind="pending",
                    level="info",
                    message=(
                        f"[INFO] Gesto mayoritario en 150 frames: {label} (cuenta={count})."
                    ),
                    label=label,
                    count=count,
                )
            )
        return events

    # ------------------------------------------------------------------
    # Acciones manuales desde teclado
    # ------------------------------------------------------------------
    def manual_add(self, label: Optional[str]) -> List[SequenceEvent]:
        if label in (None, "????"):
            return [
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message="[WARN] No hay gesto estable para añadir manualmente.",
                )
            ]
        if not self._armed:
            return [
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message="[WARN] Necesitas hacer el gesto 'demonio' antes de añadir gestos.",
                )
            ]
        if self.action_count >= self.max_actions:
            return [
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message="[WARN] Lista llena. Usa gesto 'cool' para guardar y reiniciar.",
                )
            ]

        label_added = self._append_action(label)
        self.pending_label = None
        return [
            SequenceEvent(
                kind="confirm",
                level="info",
                message=f"[INFO] Añadido gesto a la lista manualmente: {label_added}",
                label=label_added,
            ),
            SequenceEvent(
                kind="info",
                level="info",
                message=f"[INFO] Lista actual: {self._actions}",
            ),
        ]

    def manual_save(self) -> List[SequenceEvent]:
        events: List[SequenceEvent] = []
        if not self._actions:
            events.append(
                SequenceEvent(
                    kind="warn",
                    level="warn",
                    message="[WARN] No hay acciones para guardar.",
                )
            )
            return events

        events.append(
            SequenceEvent(
                kind="save",
                level="info",
                message=f"[INFO] Secuencia guardada manualmente: {self._actions}",
                actions=self.actions,
            )
        )
        self.reset()
        events.append(
            SequenceEvent(
                kind="reset",
                level="info",
                message="[INFO] Secuencia reiniciada, realiza 'demonio' para armar de nuevo.",
                reset_consensus=True,
            )
        )
        return events

    def reset(self) -> None:
        self._armed = False
        self._actions.clear()
        self.pending_label = None

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _append_action(self, label: str) -> str:
        self._actions.append(label)
        return label
