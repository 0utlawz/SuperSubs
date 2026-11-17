"""A second opinion on each reported possession, trained on external matches only.

The decoder works frame by frame. This looks at each finished possession as a whole, and asks
whether it looks like a real one: how long it is, how sure the team model was, how likely the ball
was dead, and what comes before and after it. A possession that fails is handed back to its
neighbours, which removes short fragments, wrong-team blips and false possessions in stoppages.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config

STATE = {"H": 1.0, "A": -1.0, "D": 0.0}


def segments(labels, first, last):
    """[(start, stop, label)] of constant labels in one half, stop exclusive."""
    block = labels[first:last + 1]
    starts = np.flatnonzero(np.concatenate(([True], block[1:] != block[:-1])))
    stops = np.concatenate((starts[1:], [len(block)]))
    return [(first + int(a), first + int(b), str(block[a])) for a, b in zip(starts, stops)]


def chunk_rows(labels, p_home, p_dead, out, halves):
    """(spans, X): each reported possession, and what the checker sees about it."""
    spans, rows = [], []
    for _, first, last in halves:
        parts = segments(labels, first, last)
        for index, (start, stop, label) in enumerate(parts):
            if label not in ("H", "A"):
                continue
            team = p_home[start:stop] if label == "H" else 1 - p_home[start:stop]
            dead = p_dead[start:stop]
            before = parts[index - 1] if index > 0 else None
            after = parts[index + 1] if index + 1 < len(parts) else None
            sign = STATE[label]
            rows.append([
                (stop - start) / config.FPS,
                team.mean(), team.min(), np.percentile(team, 25), np.mean(team > 0.5),
                dead.mean(), dead.max(), np.mean(dead > 0.5),
                float(np.mean(out[start:stop])),
                STATE[before[2]] * sign if before else 2.0,
                STATE[after[2]] * sign if after else 2.0,
                (before[1] - before[0]) / config.FPS if before else -1.0,
                (after[1] - after[0]) / config.FPS if after else -1.0,
            ])
            spans.append((start, stop, label, before[2] if before else None,
                          after[2] if after else None, float(dead.mean())))
    return spans, np.array(rows, dtype=np.float32).reshape(-1, 13)


def make_model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def remove(labels, spans, p_real, threshold):
    """Hand each possession the checker rejects back to whatever surrounds it."""
    out = labels.copy()
    for (start, stop, label, before, after, dead), score in zip(spans, p_real):
        if score >= threshold:
            continue
        if before == after and before is not None:
            out[start:stop] = before
        elif dead > 0.5 and "D" in (before, after):
            out[start:stop] = "D"
        elif before in ("H", "A"):
            out[start:stop] = before
        elif after in ("H", "A"):
            out[start:stop] = after
        elif before == "D" or after == "D":
            out[start:stop] = "D"
    return out
