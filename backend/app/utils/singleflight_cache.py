import threading
from functools import wraps


def singleflight_cache(func):
    """
    A thread-safe cache decorator implementing the 'singleflight' pattern.

    The singleflight pattern ensures that for any given set of arguments,
    concurrent calls to the decorated function will only result in a single
    actual execution. Other threads with the same arguments will wait for
    the first execution to complete and then receive the same result,
    rather than triggering duplicate computations.

    This is especially useful for expensive or resource-intensive operations
    where you want to avoid redundant work and prevent cache stampede.

    Example:
        @singleflight_cache
        def load_data(key):
            # expensive operation
            ...

        # In multiple threads:
        load_data('foo')  # Only one thread will actually execute the function for 'foo'
    """
    _cache = {}
    _locks = {}
    _locks_lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        if key in _cache:
            return _cache[key]
        with _locks_lock:
            lock = _locks.setdefault(key, threading.Lock())
        with lock:
            if key in _cache:
                return _cache[key]
            result = func(*args, **kwargs)
            _cache[key] = result
            return result

    return wrapper
