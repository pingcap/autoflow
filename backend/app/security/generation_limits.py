"""Shared generation admission; Redis failures never grant unmetered work."""

from dataclasses import dataclass, field
import hashlib
import hmac
import ipaddress
import json
import logging
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class GenerationLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    global_per_minute: int = Field(default=60, ge=1, le=10000, strict=True)
    global_per_day: int = Field(default=1000, ge=1, le=100000, strict=True)
    ip_per_minute: int = Field(default=20, ge=1, le=10000, strict=True)
    ip_per_day: int = Field(default=200, ge=1, le=100000, strict=True)
    user_per_minute: int = Field(default=10, ge=1, le=1000, strict=True)
    user_per_day: int = Field(default=100, ge=1, le=100000, strict=True)
    guest_per_minute: int = Field(default=3, ge=1, le=1000, strict=True)
    guest_per_day: int = Field(default=20, ge=1, le=10000, strict=True)
    global_concurrent: int = Field(default=20, ge=1, le=100, strict=True)
    ip_concurrent: int = Field(default=5, ge=1, le=100, strict=True)
    user_concurrent: int = Field(default=2, ge=1, le=20, strict=True)
    guest_concurrent: int = Field(default=1, ge=1, le=20, strict=True)
    timeout_seconds: int = Field(default=180, ge=10, le=600, strict=True)
    provider_calls_per_request: int = Field(default=16, ge=1, le=64, strict=True)


class GenerationUnavailable(RuntimeError):
    pass


class GenerationExpired(TimeoutError):
    pass


class GenerationRateLimited(RuntimeError):
    def __init__(self, retry_after):
        super().__init__("Generation limit reached")
        self.retry_after = max(1, int(retry_after))


ADMIT = """
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local limits = cjson.decode(ARGV[1])
local periods = {60000, 86400000, 60000, 86400000, 60000, 86400000}
local buckets, counts = {}, {}
for i=1,6 do
    buckets[i] = math.floor(now / periods[i])
    counts[i] = 0
    if tonumber(redis.call('HGET', KEYS[i], 'bucket')) == buckets[i] then
        counts[i] = tonumber(redis.call('HGET', KEYS[i], 'count')) or 0
    end
    if counts[i] >= limits[i] then
        return {0, math.ceil(((buckets[i]+1)*periods[i]-now)/1000)}
    end
end
for i=7,9 do
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now)
    if redis.call('ZCARD', KEYS[i]) >= limits[i] then return {0, 1} end
end
if redis.call('EXISTS', KEYS[10]) == 1 then
    return {0, math.max(1, math.ceil(redis.call('PTTL', KEYS[10])/1000))}
end
local ttl = tonumber(ARGV[2])
for i=1,6 do
    redis.call('HSET', KEYS[i], 'bucket', buckets[i], 'count', counts[i]+1)
    redis.call('PEXPIREAT', KEYS[i], (buckets[i]+1)*periods[i]+1000)
end
for i=7,9 do
    redis.call('ZADD', KEYS[i], now+ttl, ARGV[3])
    redis.call('PEXPIRE', KEYS[i], ttl+1000)
end
redis.call('SET', KEYS[10], ARGV[3], 'PX', ttl)
return {1, ttl}
"""

RELEASE = """
for i=1,3 do redis.call('ZREM', KEYS[i], ARGV[1]) end
if redis.call('GET', KEYS[4]) == ARGV[1] then redis.call('DEL', KEYS[4]) end
return 1
"""


def ip_bucket(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if isinstance(address, ipaddress.IPv6Address):
            return str(ipaddress.ip_network(f"{address}/64", strict=False))
        return str(address)
    except ValueError:
        return "unknown"


@dataclass
class GenerationLease:
    limiter: "GenerationLimiter" = field(repr=False)
    keys: list[str] = field(repr=False)
    token: str = field(repr=False)
    deadline: float
    closed: bool = False

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if self.closed or remaining <= 0:
            raise GenerationExpired("Generation deadline exceeded")
        return remaining

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.limiter.redis.eval(RELEASE, 4, *self.keys[6:], self.token)
        except RedisError as error:
            # Do not turn an already committed answer into an error. The bounded
            # lease expires if cleanup cannot reach Redis.
            logger.warning(
                "Generation lease cleanup unavailable (%s)", type(error).__name__
            )


class GenerationLimiter:
    def __init__(
        self, redis: Redis, secret: str, limits: GenerationLimits, namespace="autoflow"
    ):
        if (
            not namespace
            or len(namespace) > 32
            or not all(
                char.isascii() and (char.isalnum() or char in "-_")
                for char in namespace
            )
        ):
            raise ValueError("Invalid generation quota namespace")
        self.redis, self.secret, self.limits = redis, secret.encode(), limits
        # Keep all admission keys in one Redis Cluster hash slot.
        self.prefix = "af:gen:{" + namespace + "}:v1:"

    def fingerprint(self, kind, value):
        if not isinstance(value, str) or len(value) > 256:
            raise ValueError("Invalid generation identity")
        data = ("autoflow-generation-v1\0" + kind + ":" + value).encode()
        return hmac.new(self.secret, data, hashlib.sha256).hexdigest()[:32]

    def acquire(
        self, *, actor: str, client_ip: str, resource: str | None, authenticated: bool
    ):
        policy = self.limits
        token = uuid4().hex
        identity = self.fingerprint("actor", actor)
        peer = self.fingerprint("ip", ip_bucket(client_ip))
        target = self.fingerprint("resource", resource or "new:" + token)
        keys = [
            self.prefix + value
            for value in (
                "global:minute",
                "global:day",
                f"actor:{identity}:minute",
                f"actor:{identity}:day",
                f"ip:{peer}:minute",
                f"ip:{peer}:day",
                "global:active",
                f"actor:{identity}:active",
                f"ip:{peer}:active",
                f"resource:{target}",
            )
        ]
        limits = [
            policy.global_per_minute,
            policy.global_per_day,
            policy.user_per_minute if authenticated else policy.guest_per_minute,
            policy.user_per_day if authenticated else policy.guest_per_day,
            policy.ip_per_minute,
            policy.ip_per_day,
            policy.global_concurrent,
            policy.user_concurrent if authenticated else policy.guest_concurrent,
            policy.ip_concurrent,
        ]
        try:
            result = self.redis.eval(
                ADMIT,
                10,
                *keys,
                json.dumps(limits),
                (policy.timeout_seconds + 120) * 1000,
                token,
            )
        except RedisError:
            raise GenerationUnavailable("Generation temporarily unavailable") from None
        if result[0] != 1:
            raise GenerationRateLimited(result[1])
        return GenerationLease(
            self, keys, token, time.monotonic() + policy.timeout_seconds
        )
