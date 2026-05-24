import time
from dataclasses import dataclass

import redis


class CircuitOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    name: str
    failure_threshold: int = 5
    reset_seconds: int = 300


class RedisCircuitBreaker:
    def __init__(self, client: redis.Redis, config: CircuitBreakerConfig):
        self.client = client
        self.config = config
        self.state_key = f"circuit:{config.name}:state"
        self.failures_key = f"circuit:{config.name}:failures"
        self.opened_at_key = f"circuit:{config.name}:opened_at"

    def allow_request(self) -> bool:
        state = self.client.get(self.state_key)
        if state != "open":
            return True

        opened_at = float(self.client.get(self.opened_at_key) or 0)
        if time.time() - opened_at < self.config.reset_seconds:
            return False

        self.client.set(self.state_key, "half_open", ex=self.config.reset_seconds)
        return True

    def ensure_allowed(self) -> None:
        if not self.allow_request():
            raise CircuitOpenError(f"circuit {self.config.name} is open")

    def record_success(self) -> None:
        pipe = self.client.pipeline()
        pipe.set(self.state_key, "closed")
        pipe.delete(self.failures_key)
        pipe.delete(self.opened_at_key)
        pipe.execute()

    def record_failure(self) -> None:
        failures = self.client.incr(self.failures_key)
        self.client.expire(self.failures_key, self.config.reset_seconds)

        if failures >= self.config.failure_threshold:
            pipe = self.client.pipeline()
            pipe.set(self.state_key, "open", ex=self.config.reset_seconds)
            pipe.set(self.opened_at_key, time.time(), ex=self.config.reset_seconds)
            pipe.execute()

    def state(self) -> dict[str, str | int]:
        return {
            "state": self.client.get(self.state_key) or "closed",
            "failures": int(self.client.get(self.failures_key) or 0),
        }
