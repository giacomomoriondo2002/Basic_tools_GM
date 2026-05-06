import numpy as np
import scipy as sp



__all__ = ["convolution", "reg_deconvolution"]



def convolution(a, b, mode='full', veca=None,vecb=None):
    """
    Compute the convolution between two signals, with optional scaling
    to preserve physical units when time vectors are provided.
    Parameters
    ----------
    a : array_like
        First input signal (1D array).
    b : array_like
        Second input signal (1D array).
    mode : {'full', 'valid', 'same', 'scaled', 'full scaled'}, optional
        Type of convolution to compute:
        - 'full' (default): standard convolution without scaling.
        - 'valid': only points where signals overlap completely.
        - 'same': result has the same length as `a`.
        - 'scaled': convolution scaled by sampling step `dx`
          and cropped/padded to match the length of `a`.
        - 'full scaled': convolution scaled by sampling step `dx`
          with its natural full length, and time vector returned.
    veca : array_like, optional
        Time vector corresponding to signal `a`. Required for
        'scaled' and 'full scaled' modes.
    vecb : array_like, optional
        Time vector corresponding to signal `b`. If None,
        defaults to `veca`.
    Returns
    -------
    conv : ndarray
        Convolved signal.
        - If `mode` in {'full', 'same', 'valid'} â†’ only the convolved values.
        - If `mode='scaled'` â†’ convolved values rescaled and
          cropped/padded to match `a`.
        - If `mode='full scaled'` â†’ tuple (conv, tfull) where `tfull`
          is the reconstructed time vector.
    tfull : ndarray, optional
        Returned only if `mode='full scaled'`. Time vector of the convolution.
    Raises
    ------
    ValueError
        If 'scaled' or 'full scaled' mode is selected without time vectors.
        If input time vectors are not uniformly sampled or have different
        sampling frequencies.
        If length of time vectors does not match the signal length.
    Notes
    -----
    - Uses `scipy.signal.fftconvolve` for efficiency.
    - Scaling by `dx` ensures that convolution has consistent units
      with continuous-time convolution.
    - In 'scaled' mode, zero-padding or cropping is applied
      to align the result with `veca`.
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> t = np.linspace(0, 1, 100)
    >>> a = np.sin(2*np.pi*5*t)
    >>> b = np.ones(20)
    >>> conv = convolution(a, b, mode='scaled', veca=t)
    >>> conv_full, tconv = convolution(a, b, mode='full scaled', veca=t)
    """
     
    if mode!='scaled' and mode!='full scaled':   #Here me manage the convolution modes already present in the in numpy and scipy functions
        conv=sp.signal.fftconvolve(a,b,mode=mode)
        return conv
    if veca is None:   #we have to check for the presence of the veca array if the user has selected the scaled and full scaled modes
        raise ValueError("Vecx required for modes 'scaled' and 'full scaled' .")
    if vecb is None: #we use the same time vectors
        vecb=veca
    if len(veca) != len(a) or len(vecb)!=len(b):
        raise ValueError("time vectors length must match the vectors one.")   #we must have matching sizes clearly
        # We verify whether the time vectors are uniformly sampled or not. Allclose does that 
    dtsa = np.diff(veca)
    dtsb = np.diff(vecb)
    if not  ((np.allclose(dtsa, dtsa[0], rtol=1e-6, atol=1e-12) and np.allclose(dtsb, dtsb[0], rtol=1e-6, atol=1e-12) )and np.allclose(dtsa[0], dtsb[0], rtol=1e-6, atol=1e-12)) :
        raise ValueError("veca and vecb must be uniformally samopled and with the same frequency.")
    dx = float(dtsa[0])   #we extract the dt form dtsa
    t0full=veca[0]+vecb[0]   #we have toi define the full scaled convolution time vector bounds. The mathematical reason behind those expressions is in the module explanation notebook 
    t0max=veca[-1]+vecb[-1]
    conv=sp.signal.fftconvolve(a,b)    #this funtion uses the FFT and is much faster
    tfull=np.arange(t0full,t0max+10*dx,dx)  #to ensure no sampling error is committed we make the vector longer and then crop at the right dimensions
    tfull=tfull[0:len(conv)]
    if mode=='full scaled':  
        return conv*dx,tfull   #we scale with the differential to approximate the integral
    elif mode=='scaled':  #we adapt for the lenght of veca
        #we might need to padd the vector
        if t0full<=veca[0]:
            new0=int((veca[0]-t0full)/dx)  #no need to add points
            newmax=int(new0+len(veca))
            conv=conv[new0:newmax]*dx
        else:
            Nptoadd=int((t0full-veca[0])/dx)   #we need a o padding before the first point
            conv=np.concatenate((np.zeros(Nptoadd),conv))
            conv=conv[0:len(veca)]*dx
        return conv
def reg_deconvolution(trace,Green,tvec,*args,mode='discrete',epsilon=None):
    """
    Regularized frequency-domain deconvolution:
        Ff = Ftrace * conj(FG) / (|FG|^2 + eps)
    where the Green kernel is sampled over lags (tau), not over absolute time.
    Parameters
    ----------
    trace : (L,) array_like
        Input signal to be deconvolved.
    Green : callable
        Function `Green(tau, *args)`, causal (tau >= 0).
    tvec : (L,) array_like
        Time vector of the signal (uniformly spaced).
    *args : tuple
        Parameters for the Green function.
    mode : {'discrete', 'continuous'}, optional
        - 'discrete': no correction is applied.
        - 'continuous': divides by `dt` to compensate the integral,
        so that sum â‰ˆ integral.
    epsilon : float or None, optional
        Regularization term; if very small, it is automatically scaled
        relative to |FG|^2.
    Returns
    -------
    f : (L,) ndarray
        Deconvolved signal, same length as the input trace.
    """
    L=len(trace)
    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):    #we check if the time vector is uniformly sampled
        raise ValueError("tvec must be uniformally samopled and with the same frequency.")
    dt=dts[0]
    tau = np.arange(L) * dt    #The convolution must be evaluated on the delays: its a response signal
    g = Green(tau, *args)
    # zero-padding to reduce the cirularity effects of the deconvolution via FFT trasform
    N = 2** int(np.ceil(np.log2(2*L - 1)))  # powe of 2 >= L+L-1 to avoid circular convolution
    Ftrace=sp.fft.rfft(trace,n=N)    #we use rfft because we know that the trace is a real vector: it increases the performances of the code
    FG=sp.fft.rfft(g,n=N)
    if epsilon is None or epsilon <= 0:      #its scales with the kernel value to better adapt
        eps = 1e-12 + 1e-6* (np.abs(FG)**2).max()   #default option
    else:
        eps = 1e-12+epsilon * (np.abs(FG)**2).max()  #custom choice 
    Ff=Ftrace*np.conj(FG)/(np.abs(FG)**2+eps)     
    f=sp.fft.irfft(Ff,n=N)[0:L]     #we deconvolve and then cut the padded section
    l=1
    if mode=='discrete':
        l=1
    elif mode=='continuous':
        l=np.diff(tvec)[0]   #we have to divide by the differntial because in the continuous: conv = âˆ« g(Ï„) u(t-Ï„) dÏ„
    
    return f/l  # 

