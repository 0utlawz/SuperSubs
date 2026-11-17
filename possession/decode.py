"""Find the best sequence of states for each half: most evidence collected, fewest changes.

This is the Viterbi algorithm over three states (H, A, D) with a fixed cost per change of state.
It is what turns flickering frame-by-frame votes into possessions: a run of opposing votes only
flips the state if it is worth more than the cost of flipping, and the flip lands where the
evidence turns, not where a smoothing window happens to end.
"""

import numpy as np

from . import config
from .evidence import H, A, D, STATES


def transition_costs(team=None, to_dead=None, from_dead=None):
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = config.P_TEAM_SWITCH if team is None else team
    cost[H, D] = cost[A, D] = config.P_TO_DEAD if to_dead is None else to_dead
    cost[D, H] = cost[D, A] = config.P_FROM_DEAD if from_dead is None else from_dead
    return cost


STAY = -1


def viterbi(evidence, cost, min_dead, team_cost=None):
    """Best state path through (T, 3) evidence, with every dead-ball stretch min_dead frames+.

    Three running scores: the best path ending at frame t in H, in A, and in D having been dead
    for at least min_dead frames. A path can only go dead by taking min_dead frames of it at once,
    which is what makes a stoppage of a few frames impossible. Ties keep the current state. A half
    may start dead (the second half before kick-off) and may end in a shorter stoppage.
    team_cost, when given, replaces the cost of a change of team frame by frame.
    """
    n = len(evidence)
    k = max(1, int(min_dead))
    rows = evidence.tolist()
    dead_sum = np.concatenate(([0.0], np.cumsum(evidence[:, D]))).tolist()
    p_team, p_to, p_from = cost[H, A], cost[H, D], cost[D, H]
    team_costs = team_cost.tolist() if team_cost is not None else None

    s_h, s_a, s_d = [0.0] * n, [0.0] * n, [0.0] * n
    b_h, b_a, b_d = [H] * n, [A] * n, [STAY] * n
    s_h[0], s_a[0], s_d[0] = rows[0]

    def live_before(j):
        """(score, state) of the better live path ending at frame j."""
        return (s_h[j], H) if s_h[j] >= s_a[j] else (s_a[j], A)

    for t in range(1, n):
        e_h, e_a, e_d = rows[t]
        h, a, d = s_h[t - 1], s_a[t - 1], s_d[t - 1]
        if team_costs is not None:
            p_team = team_costs[t]

        best, source = h, H
        if a - p_team > best:
            best, source = a - p_team, A
        if d - p_from > best:
            best, source = d - p_from, D
        s_h[t], b_h[t] = best + e_h, source

        best, source = a, A
        if h - p_team > best:
            best, source = h - p_team, H
        if d - p_from > best:
            best, source = d - p_from, D
        s_a[t], b_a[t] = best + e_a, source

        # Stay dead, or arrive at k frames of dead ball that began at frame t - k + 1.
        best, source = d + e_d, STAY
        j = t - k
        if j >= 0:
            live, live_state = live_before(j)
            entered = live - p_to + dead_sum[t + 1] - dead_sum[j + 1]
            if entered > best:
                best, source = entered, live_state
        s_d[t], b_d[t] = best, source

    end_score, end_state, end_entry = s_h[-1], H, None
    if s_a[-1] > end_score:
        end_score, end_state = s_a[-1], A
    if s_d[-1] > end_score:
        end_score, end_state = s_d[-1], D
    for start in range(max(1, n - k + 1), n):  # a stoppage still running when the half ends
        live, live_state = live_before(start - 1)
        candidate = live - p_to + dead_sum[n] - dead_sum[start]
        if candidate > end_score:
            end_score, end_state, end_entry = candidate, live_state, start

    path = np.empty(n, dtype=np.int8)
    t, state = n - 1, end_state
    if end_entry is not None:
        path[end_entry:] = D
        t = end_entry - 1
    while t >= 0:
        if state == D:
            source = b_d[t]
            if source == STAY:
                path[t] = D
                t -= 1
            else:
                path[t - k + 1:t + 1] = D
                t -= k
                state = source
        else:
            path[t] = state
            state = (b_h if state == H else b_a)[t]
            t -= 1
    return path


def center_team_switches(path, evidence):
    """Move each H<->A switch to the middle of the stretch with no team evidence around it.

    When the ball vanishes with one team and reappears with the other, every switch point inside
    the gap scores the same and Viterbi picks the first. The middle is the guess with the smallest
    worst-case error.
    """
    path = path.copy()
    neutral = np.isclose(evidence[:, H], evidence[:, A])
    for t in np.flatnonzero(path[1:] != path[:-1]) + 1:
        before, after = path[t - 1], path[t]
        if {before, after} != {H, A} or not neutral[t]:
            continue
        lo = t
        while lo > 0 and neutral[lo - 1] and path[lo - 1] == before:
            lo -= 1
        hi = t
        while hi < len(path) and neutral[hi] and path[hi] == after:
            hi += 1
        middle = (lo + hi) // 2
        path[lo:middle] = before
        path[middle:hi] = after
    return path


def decode(evidence, halves, cost=None, team_cost=None):
    """Labels for every frame: H / A / D inside the given halves, X outside them."""
    labels = np.full(len(evidence), "X", dtype="<U1")
    cost = transition_costs() if cost is None else cost
    letters = np.array(list(STATES))
    for _, first, last in halves:
        block = evidence[first:last + 1]
        block_cost = None if team_cost is None else team_cost[first:last + 1]
        path = center_team_switches(viterbi(block, cost, config.MIN_DEAD_FRAMES, block_cost), block)
        labels[first:last + 1] = letters[path]
    return labels
