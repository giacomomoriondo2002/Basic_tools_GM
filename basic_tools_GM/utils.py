import numpy as np


__all__ = ["couple_sorting", "best_index_finder"]



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
        If True, sort in descending order. If False (default), sort in
        ascending order.

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
    (array([2, 5, 9]), array(['b', 'a', 'c'], dtype='<U1'))
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
        

