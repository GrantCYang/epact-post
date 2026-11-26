"""Paired observations and exact discordance tests."""

from collections import defaultdict
import math


def paired(rows, key="correct", member="gold"):
    pairs = defaultdict(dict)
    for r in rows:
        if r[member] in pairs[r["pair_id"]]:
            raise ValueError("Duplicate pair member")
        pairs[r["pair_id"]][r[member]] = r[key]
    required = {"A", "B"} if member == "member" else {"yes", "no"}
    if not pairs or any((set(v) != required for v in pairs.values())):
        raise ValueError("Incomplete pair")
    return {k: all(v.values()) for k, v in pairs.items()}


def mcnemar(first, second):
    if set(first) != set(second):
        raise ValueError("Unmatched observations")
    b = sum((first[k] and (not second[k]) for k in first))
    c = sum((second[k] and (not first[k]) for k in first))
    n = b + c
    p = (
        min(1.0, 2 * sum((math.comb(n, k) for k in range(min(b, c) + 1))) / 2**n)
        if n
        else 1.0
    )
    return {"first_only": b, "second_only": c, "p_exact_two_sided": p}
