"""Generative Spline Fields with Persistent Curve Memory."""

from .spline import SplineField, SplineGenerator, evaluate_bspline
from .memory import (
    PersistentCurveMemory,
    PersistentPointMemory,
    PersistentGaussianMemory,
)
from .metrics import (
    control_point_drift,
    curvature_deviation,
    reprojection_error,
    compute_all_metrics,
)
from .coordinates import orient_cp, orient_pts

__version__ = "0.1.0"
