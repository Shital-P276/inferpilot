"""
serving/resilience.py

Resilience layer for the InferPilot gateway (plain infra, no ML -- the learned
thresholds/anomaly work builds ON the failure traces this generates).

Components (all additive; gateway keeps working with stock settings):
  TierBreaker      per-tier circuit breaker: CLOSED -> OPEN after
                   BREAKER_FAILURE_THRESHOLD consecutive failures, HALF_OPEN
                   trial after BREAKER_OPEN_TIMEOUT_S, back to CLOSED on probe
                   success or OPEN on probe failure. Thread-safe.
  Bulkhead         per-tier concurrency semaphore; acquire(timeout) failure
                   means tier-busy (treated like tier-unavailable by caller).
  FALLBACK_ORDER   substitution chain when the chosen tier is unavailable:
                   heavy -> balanced -> fast. fast has no fallback (fail fast).
  Fault injection  FAULT_INJECTION env "tier:rate" (e.g. "heavy:1.0") makes the
                   dispatch wrapper raise simulated failures deterministically,
                   so chaos traces can be generated without docker stops.
                   Empty/unset = disabled.

Thresholds are hand-set (env-overridable). Learning them from chaos traces is
explicit follow-up work, not done here.

Prometheus: breaker_transitions_total{tier,to_state}, breaker_state{tier}
(gauge 0=closed/1=open/2=half-open), fallback_total{from_tier,to_tier}.
"""

import asyncio
import os
import random
import threading
import time
from contextlib import asynccontextmanager

from prometheus_client import Counter, Gauge

# ---------------- Config (env-overridable, hand-set) ----------------
BREAKER_FAILURE_THRESHOLD = int(os.environ.get("BREAKER_FAILURE_THRESHOLD", "5"))
BREAKER_OPEN_TIMEOUT_S = float(os.environ.get("BREAKER_OPEN_TIMEOUT_S", "30"))
BULKHEAD_LIMITS = {
    "fast": int(os.environ.get("BULKHEAD_FAST", "100")),
    "balanced": int(os.environ.get("BULKHEAD_BALANCED", "50")),
    "heavy": int(os.environ.get("BULKHEAD_HEAVY", "25")),
}
BULKHEAD_TIMEOUT_S = float(os.environ.get("BULKHEAD_TIMEOUT_S", "5"))

# Fault injection spec, parsed once: "tier:rate" or "" (disabled).
FAULT_INJECTION_RAW = os.environ.get("FAULT_INJECTION", "").strip()


def parse_fault_injection(raw):
    if not raw or ":" not in raw:
        return None, 0.0
    tier, _, rate = raw.partition(":")
    tier = tier.strip()
    try:
        rate = float(rate)
    except ValueError:
        return None, 0.0
    if tier not in ("fast", "balanced", "heavy") or not (0.0 < rate <= 1.0):
        return None, 0.0
    return tier, rate


FAULT_TIER, FAULT_RATE = parse_fault_injection(FAULT_INJECTION_RAW)

# Fallback substitution order (cheaper tiers only; fast is terminal).
FALLBACK_ORDER = {
    "heavy": ["balanced", "fast"],
    "balanced": ["fast"],
    "fast": [],
}

# ---------------- Metrics ----------------
BREAKER_TRANSITIONS = Counter(
    "breaker_transitions_total",
    "Circuit-breaker state transitions by tier and destination state",
    ["tier", "to_state"],
)
BREAKER_STATE = Gauge(
    "breaker_state",
    "Circuit-breaker state by tier (0=closed, 1=open, 2=half-open)",
    ["tier"],
)
FALLBACKS = Counter(
    "fallback_total",
    "Fallback substitutions by original tier, substitute tier, and strategy",
    ["from_tier", "to_tier", "strategy"],
)

_STATE_TO_NUM = {"closed": 0, "open": 1, "half-open": 2}


class TierBreaker:
    """Consecutive-failure circuit breaker with half-open probe. Thread-safe."""

    def __init__(self, tier, failure_threshold=BREAKER_FAILURE_THRESHOLD,
                 open_timeout_s=BREAKER_OPEN_TIMEOUT_S):
        self.tier = tier
        self.failure_threshold = failure_threshold
        self.open_timeout_s = open_timeout_s
        self._lock = threading.Lock()
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at = 0.0
        BREAKER_STATE.labels(tier=tier).set(0)

    def _set_state(self, new_state):
        self._state = new_state
        BREAKER_STATE.labels(tier=self.tier).set(_STATE_TO_NUM[new_state])
        BREAKER_TRANSITIONS.labels(tier=self.tier, to_state=new_state).inc()

    @property
    def state(self):
        with self._lock:
            if self._state == "open" and \
                    (time.monotonic() - self._opened_at) >= self.open_timeout_s:
                self._set_state("half-open")
            return self._state

    def allow_request(self):
        """True if a call may proceed (closed, or half-open probe slot)."""
        return self.state in ("closed", "half-open")

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            if self._state in ("open", "half-open"):
                self._set_state("closed")

    def record_failure(self):
        with self._lock:
            # A failed half-open probe re-opens immediately.
            if self._state == "half-open":
                self._opened_at = time.monotonic()
                self._set_state("open")
                return
            self._consecutive_failures += 1
            if self._state == "closed" and \
                    self._consecutive_failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                self._set_state("open")

    def snapshot(self):
        with self._lock:
            return {"state": self._state,
                    "consecutive_failures": self._consecutive_failures}


class Bulkhead:
    """Per-tier concurrency caps (asyncio -- never blocks the event loop).

    Use as ``async with bulkhead.slot(tier)``; raises TierUnavailable when
    the tier is saturated past BULKHEAD_TIMEOUT_S.
    """

    def __init__(self, limits=None, timeout_s=BULKHEAD_TIMEOUT_S):
        self._sems = {t: asyncio.Semaphore(n)
                      for t, n in (limits or BULKHEAD_LIMITS).items()}
        self._timeout_s = timeout_s

    @asynccontextmanager
    async def slot(self, tier):
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._sems[tier].acquire()
        except TimeoutError:
            raise TierUnavailable(f"tier {tier} saturated (bulkhead timeout)")
        try:
            yield
        finally:
            self._sems[tier].release()


class TierUnavailable(Exception):
    """Raised when a tier call fails, is breaker-blocked, bulkhead-busy, or
    fault-injected. Carries the reason for logging/metrics."""


def maybe_inject_fault(tier):
    """Raise TierUnavailable if fault injection targets this tier. Deterministic
    per configured rate; disabled when FAULT_INJECTION is unset/invalid."""
    if tier == FAULT_TIER and FAULT_RATE > 0.0 and random.random() < FAULT_RATE:
        raise TierUnavailable(f"fault-injected failure for tier={tier}")


# Module-level singletons (one set per gateway process).
BREAKERS = {t: TierBreaker(t) for t in ("fast", "balanced", "heavy")}
BULKHEAD = Bulkhead()


def fallback_chain(tier):
    """Substitute tiers to try after `tier`, cheapest-last ordering."""
    return list(FALLBACK_ORDER.get(tier, []))


def breaker_snapshot():
    return {t: b.snapshot() for t, b in BREAKERS.items()}
