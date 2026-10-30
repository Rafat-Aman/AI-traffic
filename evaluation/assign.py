"""One-to-one assignment helper shared by the metrics (global minimum-cost bipartite matching, via `lap`)."""
import numpy as np


def min_cost_assign(cost, limit=None):
    """Rows/columns matched to minimise total cost. With `limit`, pairs costing more than it stay unmatched.
    Returns a list of (row, col). `cost` is a 2-D array of finite floats."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    import lap
    if limit is None:
        _, x, _ = lap.lapjv(cost, extend_cost=True)
    else:
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=float(limit))
    return [(int(i), int(j)) for i, j in enumerate(x) if j >= 0]
