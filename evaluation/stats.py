"""Rank correlation for X4. Reuses surveillance.impact (average ranks, NaN when undefined) rather than a second
implementation: it handles ties by Pearson-on-average-ranks, which is the tie-correct Spearman."""
import math

from surveillance.impact import spearman as _spearman
from surveillance.impact import validate as _validate


def spearman(a, b):
    """Spearman rho, or None when undefined (fewer than 2 pairs / mismatched lengths / one side constant)."""
    r = _spearman(a, b)
    return None if (r is None or (isinstance(r, float) and math.isnan(r))) else float(r)


def validate(scores, human):
    """rho + 95% bootstrap interval + permutation p over pairs (surveillance.impact.validate); n<3 -> rho None."""
    return _validate(scores, human)
