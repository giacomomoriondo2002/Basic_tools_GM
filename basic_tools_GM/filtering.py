import numpy as np
import scipy as sp

from .peaks import extr_finder
from .signals import absolute_grid_sum

__all__ = ["Good_filter_index", "optimize_filter"]


def Good_filter_index(
    Bessel_filtr_trace,
    original_trace,
    tvec,
    grid_spacing=0.5,
    N_m="auto",
    skin_param="auto",
    weight=100,
):
    """
    Compute a quality index for a filtered signal.
    The index combines the distance from the original trace with a penalty for
    spurious oscillations. Lower values indicate a better filtered signal.

    Parameters
    ----------
    Bessel_filtr_trace : array_like
        Filtered signal.
    original_trace : array_like
        Original unfiltered signal.
    tvec : array_like
        Uniformly sampled time vector with the same length as the input traces.
    grid_spacing : float, optional
        Grid spacing used to measure the distance term. Default is 0.5.
    N_m : {'auto', int}, optional
        Reference number of extrema. If 'auto', it is estimated from
        `original_trace`. Default is 'auto'.
    skin_param : {'auto', float}, optional
        Softness parameter for the oscillation penalty. If 'auto', it is set
        to 5% of `N_m`. Default is 'auto'.
    weight : float, optional
        Weight multiplying the distance term. Default is 100.

    Returns
    -------
    index : float
        Quality index of the filtered signal. Lower is better.
    """
    diff = Bessel_filtr_trace - original_trace
    absgridsum = absolute_grid_sum(diff, tvec, grid_spacing=grid_spacing)
    indextr, _, _ = extr_finder(Bessel_filtr_trace)

    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):
        raise ValueError("tvec must be uniformally sampled")

    duration = float(tvec[-1] - tvec[0])
    if duration <= 0:
        raise ValueError("tvec must be stricly increasing")

    absgridsum = absgridsum / duration

    if N_m == "auto":
        indextrraw, _, _ = extr_finder(original_trace)
        N_m = len(indextrraw)

    if isinstance(skin_param, str) and skin_param == "auto":
        skin_param = 5 * N_m / 100

    s = np.log((N_m / 12) * skin_param) / np.log(7 * (N_m - skin_param) / 10)
    k = len(indextr)

    with np.errstate(over="ignore"):
        val = 5.0 * (k - (N_m - skin_param))
        exp_term = np.exp(np.clip(val, -100.0, 700.0))

    Nindex = k + exp_term + (1 / skin_param) * k**s
    return Nindex + weight * absgridsum


def optimize_filter(
    trace_to_filter,
    tvec,
    bounds,
    order=8,
    grid_spacing=0.5,
    weight=100,
    skin_param="auto",
    return_filtered=True,
):
    """
    Select the best cutoff frequency for a digital Bessel low-pass filter.
    The function scans a range of cutoff frequencies, filters the trace for
    each candidate value, evaluates the result with `Good_filter_index`, and
    returns the frequency with the lowest index.

    Parameters
    ----------
    trace_to_filter : array_like
        Signal to be filtered.
    tvec : array_like
        Uniformly sampled time vector used to infer the sampling frequency.
    bounds : tuple
        Search range for the cutoff frequency in Hz. Use `(low, high)` for a
        default step of 1 Hz, or `(low, high, step)` for a custom step.
    order : int, optional
        Order of the digital Bessel filter. Default is 8.
    grid_spacing : float, optional
        Grid spacing passed to `Good_filter_index`. Default is 0.5.
    weight : float, optional
        Weight passed to `Good_filter_index`. Default is 100.
    skin_param : {'auto', float}, optional
        Softness parameter passed to `Good_filter_index`. Default is 'auto'.
    return_filtered : bool, optional
        If True, return the best filtered signal. Default is True.

    Returns
    -------
    optfreq : float
        Cutoff frequency with the lowest quality index.
    best_index : float
        Minimum value of `Good_filter_index` found during the search.
    filtered : ndarray, optional
        Signal filtered with the optimal cutoff frequency. Returned only when
        `return_filtered=True`.
    """
    N_max = len(extr_finder(trace_to_filter)[0])

    if len(bounds) == 2:
        cut_off = np.arange(bounds[0], bounds[1] + 1, 1)
    elif len(bounds) == 3:
        cut_off = np.arange(bounds[0], bounds[1] + 1, bounds[2])
    else:
        raise ValueError("bounds must be (low, high) or (low, high, step)")

    indexv = np.zeros(len(cut_off))
    tvec = np.asarray(tvec, dtype=float)

    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):
        raise ValueError("tvec must be uniformally sampled")

    dt = float(dts[0])
    fs = 1 / dt

    for i, cutoff_freq in enumerate(cut_off):
        Wn = 2 * cutoff_freq / fs
        if not (0.0 < Wn < 1.0):
            indexv[i] = np.inf
            continue

        sos = sp.signal.bessel(N=order, Wn=Wn, btype="low", analog=False, output="sos")

        try:
            BesselFiltr_SC = sp.signal.sosfiltfilt(sos, trace_to_filter)
        except ValueError:
            try:
                BesselFiltr_SC = sp.signal.sosfiltfilt(
                    sos,
                    trace_to_filter,
                    method="gust",
                )
            except Exception:
                BesselFiltr_SC = sp.signal.sosfilt(sos, trace_to_filter)

        try:
            indexv[i] = Good_filter_index(
                BesselFiltr_SC,
                trace_to_filter,
                tvec,
                N_m=N_max,
                grid_spacing=grid_spacing,
                weight=weight,
                skin_param=skin_param,
            )
        except Exception:
            indexv[i] = np.inf

    optfreq = cut_off[indexv.argmin()]
    if not return_filtered:
        return optfreq, indexv.min()

    Wnopt = 2 * optfreq / fs
    if not (0.0 < Wnopt < 1.0):
        raise ValueError("Cutoff ottimale fuori banda digitale; rivedere bounds/fs.")

    sosopt = sp.signal.bessel(N=order, Wn=Wnopt, btype="low", analog=False, output="sos")

    try:
        BesselFiltr_SC = sp.signal.sosfiltfilt(sosopt, trace_to_filter)
    except ValueError:
        try:
            BesselFiltr_SC = sp.signal.sosfiltfilt(
                sosopt,
                trace_to_filter,
                method="gust",
            )
        except Exception:
            BesselFiltr_SC = sp.signal.sosfilt(sosopt, trace_to_filter)

    return optfreq, indexv.min(), BesselFiltr_SC
