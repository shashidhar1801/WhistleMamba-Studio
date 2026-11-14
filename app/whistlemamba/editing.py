from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from .project import AnalysisProject, Contour, ContourPoint, utc_now


@dataclass(frozen=True)
class EditRecord:
    timestamp_utc: str
    action: str
    contour_ids: tuple[str, ...]
    details: str = ""


class ContourEditor:
    """Non-destructive contour editor with bounded undo and redo stacks."""

    def __init__(self, project: AnalysisProject, *, maximum_history: int = 200):
        self.project = project
        self.maximum_history = maximum_history
        self._undo: list[tuple[list[Contour], EditRecord]] = []
        self._redo: list[tuple[list[Contour], EditRecord]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _apply(
        self,
        action: str,
        contour_ids: tuple[str, ...],
        operation: Callable[[list[Contour]], None],
        details: str = "",
    ) -> None:
        before = deepcopy(self.project.edited_contours)
        operation(self.project.edited_contours)
        record = EditRecord(utc_now(), action, contour_ids, details)
        self._undo.append((before, record))
        self._undo = self._undo[-self.maximum_history :]
        self._redo.clear()
        self.project.edit_log.append(record.__dict__)
        self.project.modified_utc = record.timestamp_utc

    def _index(self, contour_id: str) -> int:
        for index, contour in enumerate(self.project.edited_contours):
            if contour.contour_id == contour_id:
                return index
        raise KeyError(f"unknown contour: {contour_id}")

    def add_contour(self, contour: Contour) -> None:
        if any(item.contour_id == contour.contour_id for item in self.project.edited_contours):
            raise ValueError(f"duplicate contour id: {contour.contour_id}")
        self._apply("add_contour", (contour.contour_id,), lambda items: items.append(contour))

    def delete_contour(self, contour_id: str) -> None:
        def operation(items: list[Contour]) -> None:
            items.pop(self._index(contour_id))

        self._apply("delete_contour", (contour_id,), operation)

    def add_point(self, contour_id: str, point: ContourPoint) -> None:
        def operation(items: list[Contour]) -> None:
            contour = items[self._index(contour_id)]
            contour.points.append(point)
            contour.points.sort(key=lambda item: item.time_s)

        self._apply("add_point", (contour_id,), operation)

    def move_point(self, contour_id: str, point_index: int, point: ContourPoint) -> None:
        def operation(items: list[Contour]) -> None:
            contour = items[self._index(contour_id)]
            contour.points[point_index] = point
            contour.points.sort(key=lambda item: item.time_s)

        self._apply("move_point", (contour_id,), operation, f"point_index={point_index}")

    def delete_point(self, contour_id: str, point_index: int) -> None:
        def operation(items: list[Contour]) -> None:
            contour = items[self._index(contour_id)]
            if len(contour.points) == 1:
                raise ValueError("cannot delete the only point; delete the contour instead")
            contour.points.pop(point_index)

        self._apply("delete_point", (contour_id,), operation, f"point_index={point_index}")

    def set_review_state(self, contour_id: str, state: str) -> None:
        if state not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid review state: {state}")

        def operation(items: list[Contour]) -> None:
            items[self._index(contour_id)].review_state = state  # type: ignore[assignment]

        self._apply("set_review_state", (contour_id,), operation, state)

    def set_notes(self, contour_id: str, notes: str) -> None:
        def operation(items: list[Contour]) -> None:
            items[self._index(contour_id)].notes = notes

        self._apply("set_notes", (contour_id,), operation)

    def split(self, contour_id: str, point_index: int) -> tuple[str, str]:
        contour = self.project.edited_contours[self._index(contour_id)]
        if not 0 < point_index < len(contour.points):
            raise ValueError("split point must leave at least one point in each contour")
        left = Contour.create(
            contour.points[:point_index], source="manual", parent_ids=[contour_id]
        )
        right = Contour.create(
            contour.points[point_index:], source="manual", parent_ids=[contour_id]
        )

        def operation(items: list[Contour]) -> None:
            index = self._index(contour_id)
            items[index : index + 1] = [left, right]

        self._apply("split_contour", (contour_id, left.contour_id, right.contour_id), operation)
        return left.contour_id, right.contour_id

    def merge(self, first_id: str, second_id: str) -> str:
        first = self.project.edited_contours[self._index(first_id)]
        second = self.project.edited_contours[self._index(second_id)]
        merged = Contour.create(
            first.points + second.points,
            source="manual",
            parent_ids=[first_id, second_id],
        )

        def operation(items: list[Contour]) -> None:
            items[:] = [item for item in items if item.contour_id not in {first_id, second_id}]
            items.append(merged)

        self._apply("merge_contours", (first_id, second_id, merged.contour_id), operation)
        return merged.contour_id

    def undo(self) -> bool:
        if not self._undo:
            return False
        previous, record = self._undo.pop()
        self._redo.append((deepcopy(self.project.edited_contours), record))
        self.project.edited_contours = previous
        self.project.edit_log.append(
            EditRecord(utc_now(), "undo", record.contour_ids, record.action).__dict__
        )
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        following, record = self._redo.pop()
        self._undo.append((deepcopy(self.project.edited_contours), record))
        self.project.edited_contours = following
        self.project.edit_log.append(
            EditRecord(utc_now(), "redo", record.contour_ids, record.action).__dict__
        )
        return True
