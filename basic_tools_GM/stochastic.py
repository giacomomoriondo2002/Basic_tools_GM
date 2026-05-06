import numpy as np
import scipy as sp
__all__ = [
    "autocovariance_matrix_1D_system",
    "Random_jump",
    "autocovariance_function",
]
def autocovariance_matrix_1D_system(X,N_iter_axis=0,unbiased=True):
    """
    Compute the two-time autocovariance matrix of a 1D stochastic process
    from an ensemble of trajectories.
    This function estimates the autocovariance matrix:
    
        C_{ij} = < (x(t_i) - mu(t_i)) (x(t_j) - mu(t_j)) >
    
    where the average is taken over the ensemble of realizations.

    Parameters
    ----------
    X : array_like
        Input data array containing multiple realizations of the stochastic process.
        The array must contain one axis corresponding to independent trajectories
        (ensemble dimension) and one axis corresponding to time.
        Examples:
        - shape (n_traj, n_time) with N_iter_axis=0
        - shape (n_time, n_traj) with N_iter_axis=1
    N_iter_axis : int, optional (default=0)
        Axis corresponding to the ensemble of independent trajectories.
        The function internally moves this axis to the first position.
    unbiased : bool, optional (default=True)
        If True, use the unbiased estimator of the covariance (divide by N-1).
        If False, use the biased estimator (divide by N), typically used in
        Monte Carlo simulations.

    Returns
    -------
    X_corr : ndarray, shape (n_time, n_time)
        Two-time autocovariance matrix. Entry (i, j) corresponds to the
        covariance between x(t_i) and x(t_j).

    Notes
    -----
    - The function assumes that different trajectories are independent realizations
      of the same stochastic process.
    - The process does not need to be stationary: the full two-time covariance
      matrix is computed.
    - For stationary processes, the covariance depends only on the time lag,
      and further reduction to a 1D correlation function is possible.
    - The computation is fully vectorized and uses optimized linear algebra
      routines (BLAS).
    """
    X=np.asarray(X)
    N_iter_axis=int(N_iter_axis)
    X=np.moveaxis(X,N_iter_axis,0)
    X_norm=X-np.mean(X,0,keepdims=True)
    M= X_norm.shape[0]-1
    if not unbiased:
        M+=1
    X_corr=(X_norm.T@X_norm)/M
    return X_corr
def Random_jump(W,current_state,State_vec,test_W=True):
    """
    Sample the next state of a discrete-time Markov process from a given transition matrix.
    Given the current state and a transition matrix W, this function draws the next state
    according to the probability distribution defined by the corresponding column of W.

    Parameters
    ----------
    W : array_like, shape (N, N)
        Transition matrix of the Markov process. Element W[i, j] represents the probability
        of transitioning from state j to state i. Each column must sum to 1.
    current_state : scalar or array_like
        Current state of the system. If states are scalar, this should match an element
        of `State_vec`. If states are vector-valued, this should match one row of `State_vec`.
    State_vec : array_like, shape (N,) or (N, d)
        Collection of all possible states. If states are scalar, this is a 1D array of length N.
        If states are vector-valued, this is a 2D array where each row represents a state
        in a d-dimensional space.
    test_W : bool, optional (default=True)
        If True, checks whether each column of W sums to 1 (within numerical tolerance).
        Raises a ValueError if the condition is not satisfied.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'next_state_index' : int
            Index of the sampled next state in `State_vec`.
        - 'next_state' : scalar or ndarray
            The sampled next state.

    Raises
    ------
    ValueError
        If `test_W` is True and W is not a valid transition matrix (columns do not sum to 1),
        or if `current_state` is not found in `State_vec`.

    Notes
    -----
    The sampling is performed using inverse transform sampling: a uniform random number
    in [0, 1) is drawn and mapped to a state via the cumulative distribution function
    associated with the transition probabilities.
    This function supports both scalar and vector-valued states, provided that `State_vec`
    is structured consistently (each row corresponding to one state).

    Examples
    --------
    Scalar states:
    >>> W = np.array([[0.8, 0.2],
    ...               [0.2, 0.8]])
    >>> State_vec = np.array([0, 1])
    >>> Random_jump(W, 0, State_vec)
    Vector-valued states:
    >>> W = np.array([[0.5, 0.2, 0.3],
    ...               [0.3, 0.5, 0.2],
    ...               [0.2, 0.3, 0.5]])
    >>> State_vec = np.array([[0, 0],
    ...                       [1, 0],
    ...                       [0, 1]])
    >>> Random_jump(W, np.array([1, 0]), State_vec)
        """
    rng=np.random.default_rng()
    State_vec=np.asarray(State_vec)
    current_state=np.asarray(current_state)
    W=np.asarray(W)
    if test_W:
        test=np.sum(W,axis=0)
        if not np.all(np.isclose(test, 1)):
            raise ValueError('W must be a well defined trasition matrix')
    if np.size(current_state)>1:
        current_state_index=np.where(np.all(State_vec==current_state,axis=1))[0][0]  
    else:
        current_state_index=np.where(State_vec==current_state)[0][0]  
    random_number=rng.uniform(0,1)
    cumulative=np.cumsum(W[:,current_state_index])
    next_state_index=np.searchsorted(cumulative,random_number)
    next_state=State_vec[next_state_index]
    return {'next_state_index':next_state_index,'next_state':next_state}
#Here we have an autocovariance_function. in this case we have an autocovariance function evaluated on a time series and averaged in time and not on
#the reproductions of the experiment
def autocovariance_function(x,max_lag,max_lag_mode='discrete',dt=None):
    """
    Compute the (non-normalized) autocovariance function of a time series
    using an FFT-based correlation.
    The autocovariance is defined as:
        C(tau) = <x(t) x(t+tau)> - <x>^2
    where <.> denotes a temporal average over the overlapping interval.
    The average is computed in discrete form as a sum divided by the
    number of overlapping samples (N - k), so no explicit time-step
    scaling is required for uniformly sampled data.

    Parameters
    ----------
    x : array_like
        Input time series (1D array).
    max_lag : int or float
        Maximum lag for which the autocovariance is computed.
        - If `max_lag_mode='discrete'`, this is interpreted as a number
          of samples.
        - If `max_lag_mode='time'`, this is interpreted as a time value
          (same units as `dt`).
    max_lag_mode : {'discrete', 'time'}, optional
        Interpretation of `max_lag`:
        - 'discrete' (default): `max_lag` is given in samples.
        - 'time': `max_lag` is given in physical time units and converted
          to samples using `dt`.
    dt : float, optional
        Sampling time step. Required only if `max_lag_mode='time'`.

    Returns
    -------
    res : ndarray
        Autocovariance values for lags tau = 0, 1, ..., max_lag.
        The first value satisfies C(0) ~= Var(x).

    Notes
    -----
    - The correlation is computed via FFT-based convolution using
      time reversal: fftconvolve(x, x[::-1]).
    - The result corresponds to a discrete-time approximation of the
      continuous-time autocovariance for uniformly sampled data.
    - No normalization by the variance is performed. To obtain the
      normalized autocorrelation function (ACF), divide the output
      by res[0].

    Examples
    --------
    >>> C = autocovariance_function(x, max_lag=500)
    >>> C_norm = C / C[0]   # normalized autocorrelation
    """
    if isinstance(max_lag_mode,str) and max_lag_mode=='time':
        if dt==None:
            raise ValueError('dt required for "time" max_lag_mode')
        max_lag=int(np.round(max_lag/dt))
    conv=sp.signal.fftconvolve(x,x[::-1],mode='full')[len(x)-1:][:max_lag+1]
    base_line=np.mean(x)**2
    div_factor=len(x)-np.array(np.arange(0,max_lag+1,1))
    res=conv/div_factor-base_line
    return res
   
#Basic utilities functions:

