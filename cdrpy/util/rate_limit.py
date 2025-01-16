"""Rate limiting utilities to requests."""

from __future__ import annotations

import time

import typing as t

from collections import deque
from functools import wraps


F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class RateLimiter:
    def __init__(self, max_calls_per_second: int) -> None:
        self.max_calls_per_second: int = max_calls_per_second
        self.call_times: t.Deque[float] = deque()

    def acquire(self) -> None:
        current_time = time.time()

        # clean up old call times that are more than 1 second ago
        while self.call_times and self.call_times[0] < current_time - 1:
            self.call_times.popleft()

        # sleep if we have reached the limit of allowed calls per second
        if len(self.call_times) >= self.max_calls_per_second:
            time_to_wait = 1 - (current_time - self.call_times[0])
            time.sleep(time_to_wait)

        # log the current call
        self.call_times.append(time.time())


def rate_limit(max_calls_per_second: int) -> t.Callable[[F], F]:
    """"""
    limiter = RateLimiter(max_calls_per_second)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()  # enforce rate limit before executing the function
            return func(*args, **kwargs)

        return wrapper

    return decorator
