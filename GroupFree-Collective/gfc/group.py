"""GroupDescriptor, stable hash, epoch manager, token computation.

These are pure-Python; no GPU or distributed dependencies. See Section 2 of
the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass

_U64_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def stable_hash64(*chunks: object) -> int:
    """Deterministic 64-bit hash seeded from FNV offset; chunks fold through splitmix64.

    Accepts bytes, str, ints, tuples, and lists. Used for group ids and
    session tokens. No external dependency; no Python ``hash()``.
    """
    h = 0xCBF29CE484222325
    for c in chunks:
        if isinstance(c, bytes):
            data: bytes = c
            for b in data:
                h = _splitmix64(h ^ b)
        elif isinstance(c, str):
            for b in c.encode():
                h = _splitmix64(h ^ b)
        elif isinstance(c, (tuple, list)):
            for v in c:
                h = _splitmix64(h ^ (int(v) & _U64_MASK))
        else:
            h = _splitmix64(h ^ (int(c) & _U64_MASK))
    return h


def compute_token(session_nonce: int, group_id: int, epoch: int) -> int:
    """Compute the 64-bit barrier token for ``(group_id, epoch)`` in this session."""
    return stable_hash64(session_nonce, group_id, epoch)


@dataclass(frozen=True)
class GroupDescriptor:
    group_id: int
    ranks: tuple[int, ...]
    local_index: int  # ranks.index(self_global_rank); -1 if not a member
    size: int
    pool_row: int

    def __post_init__(self) -> None:
        assert self.size == len(self.ranks)
        assert -1 <= self.local_index < self.size
        if self.local_index >= 0:
            assert 0 <= self.local_index < len(self.ranks)


class _EpochManager:
    """Strictly monotonic per-group epoch counter, host-side.

    Host increments are done at submission time, not at GPU completion.
    Stream FIFO order makes this correct.
    """

    def __init__(self) -> None:
        self._epochs: dict[int, int] = {}

    def next_for_barrier(self, gid: int) -> int:
        e = self._epochs.get(gid, 0)
        self._epochs[gid] = e + 1
        return e

    def next_pair_for_collective(self, gid: int) -> tuple[int, int]:
        e = self._epochs.get(gid, 0)
        self._epochs[gid] = e + 2
        return e, e + 1

    def peek(self, gid: int) -> int:
        """Return the next epoch that would be issued for ``gid`` (no advance)."""
        return self._epochs.get(gid, 0)

    def snapshot(self) -> dict[int, int]:
        return dict(self._epochs)
