import numpy as np
__all__ = ["graham_smith_alg"]
def graham_smith_alg(mat):
    """
    Perform Gramâ€“Schmidt orthonormalization on a set of vectors arranged
    row-wise in a matrix.
    Given an input matrix whose rows represent vectors
    (v1, v2, ..., vn), this function constructs an orthonormal basis
    (e1, e2, ..., en) using the (modified) Gramâ€“Schmidt procedure.
    The function also returns the normalization factors (alphas),
    i.e. the Euclidean norms of the intermediate orthogonal vectors
    before normalization. These quantities are typically used as
    stretching factors in Lyapunov exponent computations.
    Parameters
    ----------
    mat : array_like
        Input matrix of shape (n, m), where each row represents a vector
        to be orthonormalized.
    Returns
    -------
    result : dict
        Dictionary with the following entries:
        - 'new basis' : ndarray
            Orthonormalized matrix of shape (n, m), whose rows form an
            orthonormal basis.
        - 'alphas' : ndarray
            Array of length n containing the norms of the orthogonalized
            vectors before normalization.
    Raises
    ------
    ValueError
        If a vector is found to be linearly dependent (or nearly so)
        on the previous ones, i.e. if its norm falls below a prescribed
        tolerance.
    Notes
    -----
    - Vectors are assumed to be stored along the rows of the input matrix.
    - The algorithm corresponds to the modified Gramâ€“Schmidt scheme,
      which is numerically more stable than the classical formulation.
    - This routine is intended for use in dynamical systems analysis,
      e.g. in the Benettin / Bettinâ€“Giorgilliâ€“Galgani algorithm for
      Lyapunov exponent computation.
    Examples
    --------
    >>> Q = graham_smith_alg(mat)['new basis']
    >>> alphas = graham_smith_alg(mat)['alphas']
    >>> np.allclose(Q @ Q.T, np.eye(Q.shape[0]))
    True
    """
    tol = 1e-14
    mat=np.asarray(mat,dtype=float)
    ortnorm_mat=np.zeros_like(mat,dtype=float)
    alphas=np.zeros(len(mat[:,0]))
    alphas[0]=np.sqrt(np.sum(mat[0,:]**2))
    ortnorm_mat[0,:]=mat[0,:]/alphas[0]
    for k in range(1,len(mat[:,0])):
        wkort=mat[k,:]
        for i in range(0,k):
            wkort=wkort-np.dot(ortnorm_mat[i,:],wkort)*ortnorm_mat[i,:]
        alphas[k]=np.sqrt(np.sum(wkort**2))
        if alphas[k] < tol:
            raise ValueError("Linearly dependent vectors detected")
        ortnorm_mat[k,:]=wkort/alphas[k]
    return {'new basis':ortnorm_mat,'alphas':alphas}
#this is very helpfull in many applications

