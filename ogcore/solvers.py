"""Pluggable outer-loop update rules for the TPI fixed-point solve.

``run_TPI``'s outer loop computes an implied path ``G(x)`` from the current
guess ``x`` of the macro/price series ``{r_p, r, w, p_m, BQ[, TR]}``; the
update rule maps ``(x, G(x), history) -> x_next``. The default ``"picard"``
rule is the damped step ``x_next = (1 - nu) x + nu G(x)`` -- and ``run_TPI``
keeps its original ``convex_combo`` path for ``"picard"``, so the default
behavior (and golden outputs) are unchanged.

The ``"anderson"`` rule instead uses the recent residual history
``f = G(x) - x`` to take larger, better-directed (superlinear) steps, selected
via ``p.TPI_outer_method``. On its own Anderson can overshoot a strongly
nonlinear map into infeasible regions; ``run_TPI`` guards it with a trust
region anchored to the always-feasible damped point (see ``run_TPI``).
"""

import numpy as np


class _Accelerator:
    """Base class: works on a flat, per-element-scaled iterate.

    The macro/price blocks differ in magnitude by orders (r ~ 0.05, BQ/TR
    large), which would swamp the least-squares in raw units. So each element
    is scaled by a fixed reference (captured on the first step, floored well
    away from zero) to put the whole vector in an O(1), dimensionless space.
    """

    def __init__(self):
        self._scale = None

    def update(self, x, gx):
        x = np.asarray(x, dtype=float)
        gx = np.asarray(gx, dtype=float)
        if self._scale is None:
            ref = np.abs(x)
            self._scale = np.maximum(ref, 1e-3 * (ref.max() or 1.0))
        step = self._step(x / self._scale, gx / self._scale)
        return step * self._scale

    def _step(self, x, g):
        raise NotImplementedError

    def reset(self):
        """Drop accumulated history so the next step restarts fresh (used by
        run_TPI's safety net when an accelerated step diverges). The fixed
        per-element scale is kept."""


class AndersonAccelerator(_Accelerator):
    """Anderson acceleration (type-II), limited memory ``m``, mixing ``beta``.

    x_{k+1} = x_k + beta f_k - (dX + beta dF) gamma, where f = g - x, dX/dF are
    the last ``m`` iterate/residual differences, and gamma solves the least
    squares ``min_gamma || f_k - dF gamma ||``. beta=1 is undamped; beta<1 adds
    damping for robustness far from the solution.
    """

    def __init__(self, m=5, beta=1.0):
        super().__init__()
        self.m = max(1, int(m))
        self.beta = float(beta)
        self._X = []
        self._F = []

    def _step(self, x, g):
        f = g - x
        self._X.append(x)
        self._F.append(f)
        if len(self._F) == 1:
            return x + self.beta * f
        m = min(self.m, len(self._F) - 1)
        dF = np.column_stack(
            [self._F[-i] - self._F[-i - 1] for i in range(1, m + 1)]
        )
        dX = np.column_stack(
            [self._X[-i] - self._X[-i - 1] for i in range(1, m + 1)]
        )
        gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
        x_next = x + self.beta * f - (dX + self.beta * dF) @ gamma
        if len(self._F) > self.m + 1:
            self._X.pop(0)
            self._F.pop(0)
        return x_next

    def reset(self):
        self._X = []
        self._F = []


def make_outer_updater(method, p):
    """Return an updater for ``p.TPI_outer_method`` (None for the native
    ``picard`` path, which ``run_TPI`` handles itself)."""
    method = (method or "picard").lower()
    if method == "picard":
        return None
    if method == "anderson":
        return AndersonAccelerator(
            m=int(getattr(p, "tpi_anderson_m", 5)),
            beta=float(getattr(p, "tpi_anderson_beta", 1.0)),
        )
    raise ValueError(f"unknown TPI_outer_method: {method!r}")


def _selftest():
    """Validate the accelerator math on a linear contraction fixed point,
    independent of OG-Core: Anderson should converge and beat plain Picard."""
    A = np.array([[0.6, 0.2, 0.0], [0.1, 0.5, 0.2], [0.0, 0.3, 0.7]])
    b = np.array([1.0, -2.0, 3.0])

    def gmap(x):  # contraction; fixed point solves (I - A) x = b
        return A @ x + b

    def run(updater, tol=1e-12, maxit=500):
        x = np.zeros(3)
        for k in range(1, maxit + 1):
            g = gmap(x)
            if np.max(np.abs(g - x)) < tol:
                return k
            x = (
                updater.update(x, g)
                if updater is not None
                else 0.5 * g + 0.5 * x
            )
        return maxit

    out = {
        "picard_iters": run(None),
        "anderson_iters": run(AndersonAccelerator(m=3, beta=1.0)),
    }
    assert out["anderson_iters"] < out["picard_iters"], out
    return out


if __name__ == "__main__":
    print(_selftest())
