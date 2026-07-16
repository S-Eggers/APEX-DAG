from collections.abc import Iterator
from typing import Any, Protocol


class NotebookIterator(Protocol):
    """Protocol for streaming notebooks from any data source."""

    current_index: int

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any]]]: ...

    def __next__(self) -> tuple[str, dict[str, Any]]: ...
