"""A small in-process rate limiter for the public Space.

The Space holds an API key and answers anyone who visits. Without a limit, one
person with a loop can spend the whole budget in a minute. This is deliberately
simple — a fixed window per session plus a global ceiling — because a public demo
needs a spend floor, not a distributed quota system.

State is per-process and resets when the Space sleeps. That is acceptable: the
goal is to stop casual abuse and runaway loops, not to enforce billing.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# Per visitor.
#
# Sized for a thesis defence, not for a casual visitor. A committee working
# through the system asks questions in a burst -- follow-ups, "try it without
# diacritics", "now switch to fixed chunking" -- and at ten per ten minutes the
# eleventh question is refused mid-demonstration. That happened during a scripted
# probe of the deployed Space and would have happened in the room.
#
# Raising this does not raise the spend ceiling: GLOBAL_LIMIT below is what
# actually bounds cost, and it is unchanged.
PER_SESSION_LIMIT = 30
PER_SESSION_WINDOW = 60 * 10       # seconds

# Across everyone, so a burst of visitors cannot drain the budget either
GLOBAL_LIMIT = 200
GLOBAL_WINDOW = 60 * 60


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()

    @staticmethod
    def _trim(stamps: deque[float], window: float, now: float) -> None:
        while stamps and now - stamps[0] > window:
            stamps.popleft()

    def check(self, session_id: str) -> tuple[bool, str]:
        """Return (allowed, message_in_albanian)."""
        now = time.time()
        with self._lock:
            self._trim(self._global, GLOBAL_WINDOW, now)
            if len(self._global) >= GLOBAL_LIMIT:
                return False, (
                    "Sistemi ka arritur kufirin e përdorimit për këtë orë. "
                    "Provo sërish më vonë."
                )

            stamps = self._sessions[session_id]
            self._trim(stamps, PER_SESSION_WINDOW, now)
            if len(stamps) >= PER_SESSION_LIMIT:
                wait = int((PER_SESSION_WINDOW - (now - stamps[0])) / 60) + 1
                return False, (
                    f"Ke bërë shumë pyetje. Prit rreth {wait} minuta dhe provo sërish."
                )

            stamps.append(now)
            self._global.append(now)
            return True, ""

    def stats(self) -> dict[str, int]:
        now = time.time()
        with self._lock:
            self._trim(self._global, GLOBAL_WINDOW, now)
            return {"global_last_hour": len(self._global),
                    "sessions_tracked": len(self._sessions)}


limiter = RateLimiter()
