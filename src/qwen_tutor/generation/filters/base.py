"""Filter base class and result type used by every filter in the pipeline.

Every filter is an async class with a stable ``name`` attribute and a
single ``check`` coroutine that returns a ``FilterResult``. Filters that
do not need to await anything still implement ``check`` as ``async def``
— that way the pipeline can ``await`` every filter uniformly without
caring whether a particular filter happens to hit the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Union

from qwen_tutor.schemas import DPOExample, EvaluationExample, SFTExample

FilterableExample = Union[SFTExample, DPOExample, EvaluationExample]


@dataclass(frozen=True)
class FilterResult:
    """Outcome of running a single filter against a single example."""

    passed: bool
    score: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Filter(ABC):
    """Base class for synchronous- or asynchronous-bodied filters.

    Subclasses set ``name`` as a class attribute (used for logging and
    rejection-counting). The body of ``check`` may be a pure CPU
    computation, in which case the coroutine simply returns its result;
    or it may be an LLM-judge call that does network I/O.
    """

    name: str = "filter"

    @abstractmethod
    async def check(self, example: FilterableExample) -> FilterResult:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
