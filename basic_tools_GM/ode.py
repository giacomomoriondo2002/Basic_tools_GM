import numpy as np
__all__ = ["Runge_Kutta_4"]
def Runge_Kutta_4(r0,f,dt,T):
    """
    Integrate a system of first-order ordinary differential equations
    using the classical fourth-order Rungeâ€“Kutta (RK4) method.
    Parameters
    ----------
    r0 : array_like
        Initial condition (state vector at t = 0).
    f : callable
        Right-hand side of the ODE system.
        The function must have the signature f(r) and return an array
        of the same shape as r, representing dr/dt.
    dt : float
        Time step of the integration.
    T : float
        Final integration time.
    Returns
    -------
    res : dict
        Dictionary containing:
        - 'tvec' : ndarray
            Time vector from 0 to T with constant step dt.
        - 'trajectory' : ndarray
            Array of shape (N+1, len(r0)) containing the state vector
            at each time step.
    Notes
    -----
    This implementation assumes a constant time step and does not
    perform adaptive step-size control. For stiff systems or high
    accuracy requirements, consider using adaptive integrators
    (e.g. scipy.integrate.solve_ivp).
    Examples
    --------
    >>> res = Runge_Kutta_4(r0, f, dt=0.01, T=40)
    >>> t = res['tvec']
    >>> r = res['r']
    """
    Np=int(np.round(T/dt))
    tvec=np.linspace(0,Np*dt,Np+1)
    r=np.zeros(shape=(len(tvec),len(r0)))
    r[0,:] = np.asarray(r0)
    for i in range(len(tvec)-1):
        K1=dt*f(r[i,:])
        K2=dt*f(r[i,:]+K1/2)
        K3=dt*f(r[i,:]+K2/2)
        K4=dt*f(r[i,:]+K3)
        Kt=(K1+2*K2+2*K3+K4)/6
        r[i+1,:]=r[i,:]+Kt
    res={'tvec':tvec,'trajectory':r}
    return res
    

