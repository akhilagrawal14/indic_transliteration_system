"""Single-flight request coalescing.

When several requests miss the dictionary and LRU for the *same* word at the same
time (e.g. a new legal term suddenly trends), each would otherwise run its own
model inference. `SingleFlight` ensures exactly one call runs the work; the rest
block on it and share the result. This caps duplicate inference on cold keys,
which is the moment the LRU cannot yet help.

FastAPI runs sync endpoints in a threadpool, so concurrent duplicates are on
separate threads; this class is thread-safe by design.
"""

import threading
from typing import Callable, Dict, Tuple, TypeVar

T = TypeVar("T")


class _Call:
    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = None
        self.error: BaseException | None = None


class SingleFlight:
    """Coalesce concurrent calls keyed by a string."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: Dict[str, _Call] = {}

    def do(self, key: str, fn: Callable[[], T]) -> Tuple[T, bool]:
        """Run `fn` once per in-flight `key`; return (result, was_leader).

        The leader (first caller for the key) executes `fn`; followers wait and
        receive the same result. `was_leader` lets the caller attribute metrics
        correctly: only the leader actually did the work.
        """
        with self._lock:
            call = self._calls.get(key)
            if call is None:
                call = _Call()
                self._calls[key] = call
                leader = True
            else:
                leader = False

        if leader:
            try:
                call.result = fn()
            except BaseException as exc:  # propagate to followers too
                call.error = exc
            finally:
                with self._lock:
                    self._calls.pop(key, None)
                call.event.set()
        else:
            call.event.wait()

        if call.error is not None:
            raise call.error
        return call.result, leader  # type: ignore[return-value]
