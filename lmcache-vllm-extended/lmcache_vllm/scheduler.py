"""
Task 2 request scheduler: reorder a batch so same-context requests run sequentially.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Protocol, TypeVar

SchedulerMode = Literal["baseline", "context_grouped"]

T = TypeVar("T")


class HasContextId(Protocol):
    context_id: str


# [Task2] Core scheduler: reorder a batch of requests
def schedule_batch(
    items: list[T],
    mode: SchedulerMode,
) -> list[T]:
    """
    baseline: preserve submission order.
    context_grouped: group by context_id; groups ordered by first appearance in batch.
    Within each group, preserve relative order.
    """
    if mode == "baseline" or not items:
        return list(items)  # [Task1] FIFO: no reordering

    # [Task2] context_grouped: group by context_id, preserve first-appearance order
    groups: OrderedDict[str, list[T]] = OrderedDict()
    for item in items:
        cid = item.context_id
        groups.setdefault(cid, []).append(item)
    ordered: list[T] = []
    for group in groups.values():
    # flat all the requests. Requests with the same context_id will be placed together.
        ordered.extend(group)
    return ordered


# [Task1 Q3 / Task2] Count adjacent pairs with same context_id (measures cache locality)
def adjacent_same_context_pairs(items: list[HasContextId]) -> int:
    if len(items) < 2:
        return 0
    return sum(
        1 for i in range(len(items) - 1) if items[i].context_id == items[i + 1].context_id
    )
