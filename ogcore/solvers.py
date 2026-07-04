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
    """
    Base class for outer-loop update rules: works on a flat,
    per-element-scaled iterate.

    The macro/price blocks differ in magnitude by orders (r ~ 0.05, BQ/TR
    large), which would swamp the least-squares in raw units. So each element
    is scaled by a fixed reference (captured on the first step, floored well
    away from zero) to put the whole vector in an O(1), dimensionless space.
    """

    def __init__(self):
        self._scale = None

    def update(self, x, gx):
        """
        Propose the next iterate from the current iterate and map value.

        Args:
            x (array_like): current iterate of the flattened outer-loop
                variables
            gx (array_like): value of the fixed-point map G(x) implied by
                the model at the current iterate

        Returns:
            x_next (Numpy array): proposed next iterate, in the same
                (unscaled) units as ``x``
        """
        x = np.asarray(x, dtype=float)
        gx = np.asarray(gx, dtype=float)
        if self._scale is None:
            ref = np.abs(x)
            self._scale = np.maximum(ref, 1e-3 * (ref.max() or 1.0))
        step = self._step(x / self._scale, gx / self._scale)
        return step * self._scale

    def _step(self, x, g):
        """
        Compute the next iterate in scaled units (implemented by
        subclasses).

        Args:
            x (Numpy array): current iterate, per-element scaled
            g (Numpy array): fixed-point map value at ``x``, per-element
                scaled

        Returns:
            x_next (Numpy array): next iterate, per-element scaled
        """
        raise NotImplementedError

    def reset(self):
        """
        Drop accumulated history so the next step restarts fresh (used by
        ``run_TPI``'s safety net when an accelerated step diverges). The
        fixed per-element scale is kept.

        Returns:
            None
        """


class AndersonAccelerator(_Accelerator):
    r"""
    Anderson acceleration (type-II) with limited memory for the TPI outer
    loop.

    Given the residual :math:`f_k = G(x_k) - x_k` and the differences
    :math:`\Delta X, \Delta F` of the last ``m`` iterates and residuals,
    the update is

    .. math::
        x_{k+1} = x_k + \beta f_k - (\Delta X + \beta\Delta F)\gamma,

    where :math:`\gamma` solves the least squares problem
    :math:`\min_{\gamma}\ \lVert f_k - \Delta F\gamma\rVert`.
    ``beta = 1`` is undamped; ``beta < 1`` adds damping for robustness far
    from the solution.

    Args:
        m (int): number of previous iterates kept in the acceleration
            memory
        beta (float): mixing (relaxation) parameter applied to the
            residual
    """

    def __init__(self, m=5, beta=1.0):
        super().__init__()
        self.m = max(1, int(m))
        self.beta = float(beta)
        self._X = []
        self._F = []

    def _step(self, x, g):
        """
        Compute the Anderson-accelerated next iterate in scaled units.

        Args:
            x (Numpy array): current iterate, per-element scaled
            g (Numpy array): fixed-point map value at ``x``, per-element
                scaled

        Returns:
            x_next (Numpy array): accelerated next iterate, per-element
                scaled
        """
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
        """
        Clear the stored iterate and residual history so the next step
        restarts the acceleration from scratch. The fixed per-element
        scale is kept.

        Returns:
            None
        """
        self._X = []
        self._F = []


def make_outer_updater(method, p):
    """
    Create the outer-loop updater selected by ``p.TPI_outer_method``.

    Args:
        method (str or None): outer-loop update rule, either "picard" or
            "anderson" (None defaults to "picard")
        p (OG-Core Specifications object): model parameters

    Returns:
        updater (AndersonAccelerator or None): accelerator instance for
            "anderson", or None for "picard", in which case ``run_TPI``
            keeps its native damped update

    Raises:
        ValueError: if ``method`` is not a recognized update rule
    """
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
    """
    Validate the accelerator math on a linear contraction fixed point,
    independent of OG-Core: Anderson should converge and beat plain
    Picard.

    Returns:
        out (dict): iterations to convergence for the damped Picard and
            Anderson update rules
    """
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
