"""Tests for SingleFlight concurrency coalescing."""

import threading
import time

from server.singleflight import SingleFlight


def test_concurrent_same_key_runs_once():
    """Many threads hitting the same key run the work function exactly once."""
    sf = SingleFlight()
    calls = {"n": 0}
    started = threading.Event()

    def work():
        calls["n"] += 1
        started.set()
        time.sleep(0.05)  # hold the key so others pile up behind the leader
        return "value"

    results = []
    leaders = []

    def caller():
        r, leader = sf.do("k", work)
        results.append(r)
        leaders.append(leader)

    threads = [threading.Thread(target=caller) for _ in range(10)]
    threads[0].start()
    started.wait(1.0)  # ensure thread 0 is the leader and is mid-flight
    for t in threads[1:]:
        t.start()
    for t in threads:
        t.join(2.0)

    assert calls["n"] == 1                 # only one inference ran
    assert results == ["value"] * 10       # everyone got the result
    assert sum(leaders) == 1               # exactly one leader


def test_different_keys_run_independently():
    sf = SingleFlight()
    a, a_leader = sf.do("a", lambda: 1)
    b, b_leader = sf.do("b", lambda: 2)
    assert (a, b) == (1, 2)
    assert a_leader and b_leader           # distinct keys are each their own leader


def test_error_propagates_and_clears():
    sf = SingleFlight()

    def boom():
        raise ValueError("nope")

    try:
        sf.do("k", boom)
        assert False, "should have raised"
    except ValueError:
        pass
    # Key is cleared after failure, so a later call re-runs cleanly.
    assert sf.do("k", lambda: 42) == (42, True)
