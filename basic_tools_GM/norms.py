import numpy as np

__all__ = ["gen_Lp_norm", "gen_Hn_norm"]

def gen_Lp_norm(f, interval, p=2, n_gauss=12, cuts=()):
    """
    Compute the Lp norm of a function on a one-dimensional interval.

    Parameters
    ----------
    f : callable
        Function to integrate.
    interval : array_like
        Ordered partition nodes defining the integration domain.
    p : float, optional
        Exponent of the Lp norm. Must satisfy 1 <= p < infinity.
        Default is 2.
    n_gauss : int, optional
        Number of Gauss-Legendre quadrature points per subinterval.
        Default is 12.
    cuts : iterable of float, optional
        Additional integration splitting points, useful for
        piecewise-defined functions or reduced regularity.
        Default is an empty tuple.

    Returns
    -------
    float
        Numerical approximation of the Lp norm.
    """
    if not np.isfinite(p) or p < 1:
        raise ValueError("p must satisfy 1 <= p < infinity.")

    if n_gauss < 1:
        raise ValueError("n_gauss must be a positive integer.")

    interval = np.asarray(interval, dtype=float)

    if interval.ndim != 1 or len(interval) < 2:
        raise ValueError("interval must contain at least two points.")

    if np.any(np.diff(interval) <= 0):
        raise ValueError("interval nodes must be strictly increasing.")

    z, w = np.polynomial.legendre.leggauss(n_gauss)
    accumulo = 0.0

    for a, b in zip(interval[:-1], interval[1:]):
        estremi = [a, *sorted(t for t in cuts if a < t < b), b]

        for aa, bb in zip(estremi[:-1], estremi[1:]):
            xq = (aa + bb)/2 + (bb - aa)/2 * z

            values = f(xq)

            accumulo += (bb - aa)/2 * np.dot(
                w, np.abs(values)**p
            )

    return accumulo**(1/p)



def gen_Hn_norm(functions, interval, n_gauss=12, cuts=()):
    """
    Compute the Sobolev H^n norm of a function on a 1D interval.

    Parameters
    ----------
    functions : sequence of callable
        Ordered sequence of the function and its derivatives:

            [f, df, d2f, ..., dnf]

        The Sobolev order is n = len(functions) - 1.
    interval : array_like
        Ordered partition nodes defining the integration domain.
    n_gauss : int, optional
        Number of Gauss-Legendre quadrature points.
        Default is 12.
    cuts : iterable of float, optional
        Additional splitting points used during integration.
        Default is an empty tuple.

    Returns
    -------
    float
        Numerical approximation of the H^n norm.

    Notes
    -----
    The norm is computed as:

        ||f||_Hn = sqrt(sum_k ||f^(k)||_L2**2)

    For n >= 2, piecewise polynomial functions with derivative
    discontinuities may yield a broken Sobolev norm rather than
    the global H^n norm.
    """
    if len(functions) == 0:
        raise ValueError("At least one function must be provided.")

    norm_sq = 0.0

    for func in functions:
        L2 = gen_Lp_norm(
            func,
            interval,
            p=2,
            n_gauss=n_gauss,
            cuts=cuts
        )

        norm_sq += L2**2

    return np.sqrt(norm_sq)