import numpy as np
import scipy as sp
import datetime as datetime

def autocorr_matrix_1D_system(X,N_iter_axis=0,unbiased=True):
    """
    Compute the two-time autocovariance matrix of a 1D stochastic process
    from an ensemble of trajectories.

    This function estimates the autocovariance matrix:
    
        C_{ij} = < (x(t_i) - μ(t_i)) (x(t_j) - μ(t_j)) >
    
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
   



def txt_file_writer(filename,names_of_arrays,*arrays,digit_of_precision='all',mode='f',date=True,log_file=None,delimiter='\t',auto_ordering=True):
    """
    Write multiple arrays as aligned columns into a text file, with optional 
    padding, precision control, and metadata.

    This function takes a set of arrays (possibly with different lengths), 
    pads them with NaNs up to the maximum length, and writes them as 
    tabular columns into a text file. NaN values are written as empty cells 
    to keep columns aligned. A header row with column names is included, 
    and optional metadata (date/time and description) can be written.

    Parameters
    ----------
    filename : str
        Path to the output text file.
    names_of_arrays : sequence of str
        List of column names (one name per array).
    *arrays : array-like
        Arrays to be written as columns. Arrays may have different lengths; 
        shorter arrays will be padded with NaN (rendered as empty cells).
    digit_of_precision : int or 'all', optional
        Number of digits for numeric formatting. If 'all' (default), 
        17 significant digits are written (full float64 precision).
    mode : {'f', 'g', 'e'}, optional
        Format mode for numeric output:
        - 'f': fixed-point notation (default)
        - 'g': significant digits
        - 'e': scientific notation
        If `digit_of_precision='all'`, 'g' is enforced.
    date : bool, optional
        If True (default), write current date and time at the beginning of the file.
    log_file : str or None, optional
        Optional descriptive text to include in the file header. Default is None.
    delimiter : str, optional
        Column separator string. Default is a tab ('\\t').
    auto_ordering: bool {True}
        if set on 'True' it will write the arrays in decrasing order in length; if set on 'False' it will respect the names_of_arrays oder
    Returns
    -------
    bool
        True if writing was successful.

    Notes
    -----
    - All arrays are cast to float before writing, to support NaN padding.
    - NaN values are rendered as empty cells, preserving table alignment.
    - Useful for exporting data tables with columns of unequal length.
    - Couple sorting function needed in the same module

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.2345, 2.3456, 3.4567])
    >>> b = np.array([10, 20])
    >>> txt_file_writer("output.txt", ["col1", "col2"], a, b,
    ...                 digit_of_precision=4, mode='f')
    # Produces a file:
    # date and hour: 2025-10-04 10:00:00.123456
    #
    # col1    col2
    #
    # 1.2345  10.0000
    # 2.3456  20.0000
    # 3.4567
    """
    if len(names_of_arrays) != len(arrays):
        raise ValueError("names_of_arrays must cover all of the arrays")
    if isinstance(digit_of_precision,str) and  digit_of_precision== 'all':
        digit_of_precision = 17
        mode = 'g'
    
    if mode not in ('f', 'g', 'e'):
        raise ValueError("mode must be one of those options {'f','g','e'}.")
    fmt = f"{digit_of_precision}{mode}"
    len_of_input=np.array([len(x) for x in arrays])
    if auto_ordering:
        _,arrays=couple_sorting(len_of_input,arrays,descending=True)
        _,names_of_arrays=couple_sorting(len_of_input,names_of_arrays,descending=True)



    with open(filename,'w',encoding='utf-8') as file:
        if date:
            file.write(f'date and hour: {datetime.now()}\n')
        if isinstance(log_file,str):
            file.write('\nData description: '+log_file+'\n')
        if date or log_file:
                file.write('\n')
        for i,x in enumerate(names_of_arrays):
            if i==0:
                file.write(x)
            else:
                file.write(delimiter+x)
        file.write('\n\n')
        newarrays=[]
        for arr in arrays:
            len_pad=np.max(len_of_input)-len(arr)
            arr=np.asarray(arr,dtype=float)
            arr=np.pad(arr,(0,len_pad),mode='constant',constant_values=np.nan)
            newarrays.append(arr)
        

        for arrvec in zip(*newarrays):
            for i,y in enumerate(arrvec):
                if not np.isnan(y):
                    if i==0:
                        file.write(f'{y:.{fmt}}')
                    else:
                        file.write(f'{delimiter}{y:.{fmt}}')
                else:
                        file.write(delimiter+'\n',)
            file.write('\n')


    return True




def couple_sorting(x, y, descending=False):
    """
    Sort two arrays (x, y) in parallel according to the order of x.

    Parameters
    ----------
    x : array-like
        Reference array used to determine sorting order.
    y : array-like
        Array whose elements are reordered following x.
    descending : bool, optional
        If True (default), sorts in ascending order; otherwise descending.

    Returns
    -------
    x_sorted : ndarray
        Sorted version of x.
    y_sorted : list or ndarray
        Reordered version of y, matching the order of x.

    Examples
    --------
    >>> x = [5, 2, 9]
    >>> y = ['a', 'b', 'c']
    >>> couple_sorting(x, y)
    (array([9, 5, 2]), ['c', 'a', 'b'])
    """
    x = np.asarray(x)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # get sorting indices
    indices = np.argsort(x)
    if descending:
        indices = indices[::-1]  #all of the indices, but step is -1

    # reorder x
    x_sorted = x[indices]
    try:    #if y is a numeric vector
        y_sorted = np.asarray(y)[indices]
    except Exception:    #in case it's not a numeric array
        y_sorted = [y[i] for i in indices]

    return x_sorted, y_sorted

def best_index_finder(vec,*args,toll='auto',return_vec='best',mode='strict',ignore_length_warning=False):
    """
    Find indices in a 1D pool vector corresponding to one or more target values.

    For each target value, the function searches for the closest element
    in `vec` or, alternatively, for all elements within a given tolerance.

    Parameters
    ----------
    vec : array_like
        One-dimensional array representing the pool of available values.
    *args : float
        One or more target values to be matched against `vec`.
    toll : float or {'auto'}, optional
        Tolerance for matching.
        If 'auto', the tolerance is set to the mean absolute difference
        between consecutive elements of `vec`.
    return_vec : {'best', 'all'}, optional
        Controls the type of output:
        - 'best' (default): return the index of the closest value in `vec`
          for each target.
        - 'all': return all indices in `vec` whose values are within the
          specified tolerance from each target.
    mode : {'strict', 'only close points', 'loose'}, optional
        Matching policy when `return_vec='best'`:
        - 'strict': raise a ValueError if the closest point is farther than
          `toll` from any target.
        - 'only close points': discard targets whose closest point is
          farther than `toll`.
        - 'loose': always return the closest point, regardless of distance.

    Returns
    -------
    int or ndarray or dict
        If `return_vec='best'`:
        - a single integer index if one target is provided,
        - a NumPy array of indices if multiple targets are provided.

        If `return_vec='all'`:
        - for a single target, a NumPy array of all matching indices,
        - for multiple targets, a dictionary mapping each target index
          (``target:i``) to a NumPy array of matching indices in `vec`.

    Raises
    ------
    ValueError
        If input dimensions are invalid, an unknown mode is specified,
        or if `mode='strict'` and at least one target has no match within
        the specified tolerance.

    Notes
    -----
    - The function operates only on one-dimensional arrays.
    - Broadcasting is used internally to efficiently compute distances
      between pool values and multiple targets.
    - When `mode='only close points'`, the correspondence between input
      targets and returned indices may be reduced, as non-matching targets
      are omitted.
    - No sorting of `vec` is performed; tolerance estimation with
      ``toll='auto'`` assumes approximately uniform spacing.

    Examples
    --------
    >>> vec = np.array([0.0, 0.1, 0.2, 0.3])
    >>> best_index_finder(vec, 0.12)
    1

    >>> best_index_finder(vec, 0.12, 0.28)
    array([1, 3])

    >>> best_index_finder(vec, 0.12, return_vec='all', toll=0.05)
    array([1])

    >>> best_index_finder(vec, 0.12, 0.28, return_vec='all', toll=0.05)
    {'target:0': array([1]), 'target:1': array([3])}
    """
    args=np.asarray(args)
    if isinstance(vec,(list,tuple)):    #we need to operate with arrays
        vec=np.asarray(vec)
    if mode not in ['loose','strict','only close points']:
        raise ValueError('undefined tollerance mode selected; check for a typo')
    if vec.ndim != 1:
        raise ValueError("the pool vector and the targets array must be one-dimensional.")
    if isinstance(toll,str) and toll=='auto':
        toll=np.mean(np.abs(np.diff(vec)))
    if len(vec)<len(args) and ignore_length_warning:
        print('Warning best_index_finder: len(target vector)>len(pool vector)')
    if len(args)==0 or len(vec)==0:
        raise ValueError('pool vector and targets array must have at least one element')
    if isinstance(return_vec,str) and return_vec=='best':
        if len(args)>1:
            indvec=np.zeros(len(args),dtype=int)
            diffmat=np.abs(vec[None, :] - args[:, None])
            indvec=np.argmin(diffmat,axis=1)
            if isinstance(mode,str) and mode=='only close points':
                indvec=np.delete(indvec,np.nonzero(np.abs(vec[indvec]-args)>toll)[0])
            if isinstance(mode,str) and mode=='strict':
                if len(np.nonzero(np.abs(vec[indvec]-args)>toll)[0])>0:
                    raise ValueError(f'target poits {np.nonzero(np.abs(vec[indvec]-args)>toll)[0]} not close enough to any point in the pool vector')
            return indvec

        else:
            try: 
                return np.nonzero(vec==args[0])[0][0]
            except:
                if args[0]>np.max(vec)+toll or args[0]<np.min(vec)-toll:
                        raise ValueError(f'target point {args[0]} is not present in the pool vector')
                elif (isinstance(mode,str) and mode=='strict'):
                        if np.min(np.abs(vec-args[0]))>toll:
                            raise ValueError(f'target point {args[0]} is not close enough to any point in the pool vector')
                        else:
                            return np.argmin(np.abs(vec-args[0]))
                elif isinstance(mode,str) and mode=='loose':
                    return np.argmin(np.abs(vec-args[0]))
    elif isinstance(return_vec,str) and return_vec=='all':
        if len(args)>1:
            diffmat=np.abs(vec[None, :] - args[:, None])
            indvec=np.nonzero(diffmat<=toll)
            res={}
            for i in range(len(args)):
                key=f'target:{i}'
                index=np.nonzero(indvec[0]==i)[0]
                res.update({key:indvec[1][index]})
            return res

        else:
            return np.nonzero(np.abs(vec-args[0])<=toll)[0]