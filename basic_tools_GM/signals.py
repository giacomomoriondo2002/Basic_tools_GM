import numpy as np
from .utils import best_index_finder
from .peaks import extr_finder
__all__ = ["patcher", "Interp_grid", "absolute_grid_sum"]
def patcher(trace,tvec,d='auto',atol=1e-10):
    """
    Extends (patches) the trace at the beginning and end with squared half-sine waves,
    matched in value and slope at the boundaries, to stabilize FFTs/deconvolutions.
    Parameters
    ----------
    trace : (N,) array_like
        Input signal.
    tvec : (N,) array_like
        Time vector (uniformly sampled).
    d : float or {'auto'}
        Half-period (s) of the base sine wave. If 'auto', the signal duration is used.
    atol : float
        Tolerance threshold for â€œalmost zeroâ€ tests.
    Returns
    -------
    tnew : ndarray
        New time vector with patch applied.
    patched_trace : ndarray
        Signal with patches at the beginning and end.
    t0, tf : float
        Limits of the original segment within (tnew, patched_trace).
    """
    if trace.ndim != 1 or tvec.ndim != 1 or trace.size != tvec.size:   #input check
        raise ValueError("trace and tvec must be 1D and have the same length.")
    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):   #sample uniformity check
        raise ValueError("tvec must be uniformly sampled.")
    dt = float(dts[0])
    fs = 1/ dt
    if isinstance(d,str) and d=='auto':    #patch lenght settings: d can be either custo or auto (lenght in seconds)
        d=len(trace)*dt     #the same lenght of the trace: this is found to work well 
    elif d<=0:
        raise ValueError('d must be positive')   #d input check
    else: 
        d=d
    #back patch
    af=trace[-1]
    bf=np.diff(trace)[-1]*fs  #a and b are the value of the function and of the diff vector in the first point, respectly. It's very important that the values are exatly those or the deconvolution will diverge
    #front patch
    a0=trace[0]
    b0=np.diff(trace)[0]*fs  
    if a0>atol and af>atol:    #if we have to patch both sides
        K0=a0+(d*b0/np.pi)**2/a0   #K0 and epsilon0 are exitemated from those expressions
        arg0=(2*d*b0)/(np.pi*K0)
        arg0=np.clip(arg0,-1,1)
        epsilon0=(d/np.pi)*np.arcsin(arg0)
        test00=K0*np.sin((np.pi/(2*d))*(d-epsilon0))**2  #There are two possibles expressions due to the simmetry of the sine function (the actual solution of sin(x)=y is arcsin(x) and pi-arcsin(x)). To determine wich one is the appropriate one for the given variables we physically check.
        test01=K0*np.sin((np.pi/(2*d))*epsilon0)**2   
        if np.abs(trace[0]-test00)<np.abs(trace[0]-test01):
            P0=d-epsilon0    #if the first one is the right choiche then the point of contact between the trace and the patch (P0) is this one
        else:
            P0=2*d+epsilon0    #in the other case is this one
        l0=P0   #l0 is the real lenght of the patch. It can be shortsr or much longer than d depending on the different conditions of the trace to patch
        Kf=af+(d*bf/np.pi)**2/af    
        argf=(2*d*bf)/(np.pi*Kf)
        argf=np.clip(argf,-1,1)
        epsilonf=(d/np.pi)*np.arcsin(argf)
        test10=Kf*np.sin((np.pi/(2*d))*(d-epsilonf))**2  #we do the same for the ending part
        test11=Kf*np.sin((np.pi/(2*d))*epsilonf)**2
        if np.abs(trace[-1]-test10)<np.abs(trace[-1]-test11):
            Pf=d-epsilonf
        else:
            Pf=epsilonf
        
        lf=2*d-Pf  #lenght of the ending patch time vector in seconds.  
        t_patch0=np.arange(0,l0+dt,dt)    #this is the front patch tvector
        t_patch0=t_patch0-(t_patch0[-1]-P0)
        t_patchf=np.arange(Pf,Pf+lf+dt,dt)   #ending part 
        t_patchf=t_patchf-(t_patchf[0]-Pf)
        tnew=np.concatenate([t_patch0[:-1],tvec+P0,t_patchf[1:]-Pf+P0+tvec[-1]])   #new time vector
        patched_trace=np.zeros(len(tnew))    
        patched_trace[0:len(t_patch0)]=K0*np.sin((np.pi/(2*d))*t_patch0)**2   #here we insert the pieces in the patched trace vector
        patched_trace[len(t_patch0)-1:len(tvec)+len(t_patch0)-1]=trace
        patched_trace[len(tvec)+len(t_patch0)-2:]=Kf*np.sin((np.pi/(2*d))*t_patchf)**2
        t0=l0    #and we generate the time stamps to isolate the trace later in the code
        tf=l0+tvec[-1]
    elif a0<=atol and af>atol:   #case with only the final patch to be made. Works exactly as shown above
        Kf=af+((d*bf/np.pi)**2)/af
        arg=(2*d*bf)/(np.pi*Kf)
        arg=np.clip(arg,-1,1)
        epsilonf=(d/np.pi)*np.arcsin(arg)
        test10=Kf*np.sin((np.pi/(2*d))*(d-epsilonf))**2  
        test11=Kf*np.sin((np.pi/(2*d))*epsilonf)**2
        if np.abs(trace[-1]-test10)<np.abs(trace[-1]-test11):
            Pf=d-epsilonf
        else:
            Pf=epsilonf
        
        lf=2*d-Pf
        t_patchf=np.arange(Pf,Pf+lf+dt,dt)
        t_patchf=t_patchf-(t_patchf[0]-Pf)
        tnew=np.concatenate([tvec,t_patchf[1:]-Pf+tvec[-1]])
        patched_trace=np.zeros(len(tnew))
        patched_trace[0:len(tvec)]=trace
        patched_trace[len(tvec)-1:]=Kf*(np.sin((np.pi)/(2*d)*t_patchf))**2
        t0=0
        tf=tnew[len(tvec)-1]
    elif a0>atol and af<=atol:  #case with only the fornt patch to be made. Works exactly as shown above
        K0=a0+((d*b0/np.pi)**2)/a0
        arg=(2*d*b0)/(np.pi*K0)
        arg=np.clip(arg,-1,1)
        epsilon0=(d/np.pi)*np.arcsin(arg)
        test00=K0*np.sin((np.pi/(2*d))*(d-epsilon0))**2  #sarebbe il coseno
        test01=K0*np.sin((np.pi/(2*d))*epsilon0)**2
        if np.abs(trace[0]-test00)<np.abs(trace[0]-test01):
            P0=d-epsilon0
        else:
            P0=2*d+epsilon0
        
        l0=P0
        t_patch0=np.arange(0,l0+dt,dt)
        t_patch0=t_patch0-(t_patch0[-1]-P0)
        tnew=np.concatenate([t_patch0[:-1],tvec+P0])
        patched_trace=np.zeros(len(tnew))
        patched_trace[len(t_patch0):]=trace
        patched_trace[0:len(t_patch0)]=K0*(np.sin((np.pi)/(2*d)*t_patch0))**2
        t0=tnew[len(t_patch0)]
        tf=tnew[-1]
    else: #there can also be a trace that reuires no patching. It's better to adpat the functiom to this eventuality since it could be used inside cycles
        tnew = tvec.copy()
        patched_trace = trace.copy()
        t0, tf = tvec[0], tvec[-1]
    # Guard against boundary-rounding artifacts that can create one off-grid jump
    # at the patch/trace junction after cropping or ROI selection.
    if len(tnew) > 1:
        dts_new = np.diff(tnew)
        if not np.allclose(dts_new, dts_new[0], rtol=1e-6, atol=1e-12):
            tnew = tnew[0] + np.arange(len(tnew)) * dt
            start_idx, stop_idx = best_index_finder(tnew, t0, tf, mode='loose')
            t0 = float(tnew[start_idx])
            tf = float(tnew[stop_idx])
    return tnew,patched_trace,t0,tf    #this is the tuple the function gives out
#Those function are needed in one needs to find maxima of data sets
def Interp_grid(trace,Interpeak_mask,tvec,grid_spacing=5,correct_for_mins_duplication=False,minima_indices=None):   #questa funzione fa il suo dovere
    """
    Builds a regular time grid and, for each grid node, computes the average value
    of the signal in adjacent windows using only the points marked True by a mask
    (e.g., â€œinter-peakâ€ portions).
    Parameters
    ----------
    trace : array_like, shape (N,)
        Uniformly sampled signal.
    Interpeak_mask : array_like of bool, shape (N,)
        Boolean mask (True = sample usable for interpolation). Must have the same
        length as `trace`.
    tvec : array_like, shape (N,)
        Monotonically increasing, uniformly spaced time vector.
    grid_spacing : float, optional
        Grid spacing in seconds. Grid nodes are placed at multiples of
        `grid_spacing` (excluding the very start) up to just before the end.
        Default is 5.
    correct_for_mins_duplication : bool, optional
        If True, enable correction for duplicated minima in the interpolation grid.
        Default is False.
    minima_index : array_like of int or None, optional
        Indices of minima to use for the duplication-correction logic. If None,
        they are computed internally (less efficient during optimization). Default is None.
    Returns
    -------
    tgrid : ndarray, shape (M,)
        Grid node times for which an average could be computed.
    points_to_interpolate : ndarray, shape (M,)
        Window-averaged signal values around each grid node, computed using only
        samples where `Interpeak_mask` is True.
    Notes
    -----
    - The window associated with node i is [grid[i-1], grid[i+1]]. At the edges,
    [tvec[0], grid[1]] or [grid[-2], tvec[-1]] is used.
    - If a window contains no True samples in the mask, that node is discarded
    (NaN is produced and removed before returning).
    - Uniform sampling is assumed: `np.diff(tvec)` must be constant.
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 20, 201)
    >>> x = np.sin(t) + 0.1 * np.random.randn(t.size)
    >>> mask = x < 0.5   # e.g., keep only sub-threshold values
    >>> tg, yi = Interp_grid(x, mask, t, grid_spacing=2)
    >>> tg.shape == yi.shape
    True
    """
    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):   #this is the usual sample uniformity control
        raise ValueError("tvec deve essere uniformemente campionato.")
    dt = float(dts[0])
    fs =1/ dt
    Iterpgrid=[i for i in range(int(grid_spacing*fs),len(tvec),int(grid_spacing*fs))]  #grid point mask
    if len(Iterpgrid)==0:
        raise ValueError('Interpolation grid is empty; check grid_spacing or Interpeak_mask')  #we check whther we can procede
    tgrid=tvec[Iterpgrid]  #we build the time grid
    point_to_inter=np.zeros(len(Iterpgrid))  #space preallocation
    for i in range(len(Iterpgrid)): #we have to build the grid y values
        if i==0:   #if it's the first point the area of summation starts from 0
            maskinter=range(0,Iterpgrid[i+1])
        elif i==len(Iterpgrid)-1:   #or ends with the last trace point if we area dealing with the last point 
            maskinter=range(Iterpgrid[i-1],len(trace))
        else:   
            maskinter=range(Iterpgrid[i-1],Iterpgrid[i+1])   #we mediate all the points in from the point before to the one after: that is the value of the y
        if len(trace[maskinter][Interpeak_mask[maskinter]])>0:   #if we have points to mediate
            point_to_inter[i]=np.mean(trace[maskinter][Interpeak_mask[maskinter]])  #we are taking the points that are inside the reagion for the i-th point and that are flagged by the interpreak mask 
        else:
            point_to_inter[i]=np.nan   #if we have no point: the value is nan
    
    mask=np.isnan(point_to_inter)   #and we procede eliminating the nan
    tgrid=tgrid[~mask]   #now we have the t values of the grid
    point_to_inter=point_to_inter[~mask]   # and the y's
    #When the minima have been included in the interpeak musk there is very common phenomena called 'minima duplication', occuring when a minimum is isolated and duplicated by the grid because two points of the grid have it in their respective window.
    # This part corrects the output to avoid such a duplication
    if correct_for_mins_duplication:   
        difftgrid=np.diff(tgrid)
        if minima_indices is None:              #ottimizzazione
            _,_,mins=extr_finder(trace)
        else: 
            mins=minima_indices
        minspos=tvec[mins]
        index_of_mins_in_tgrid=best_index_finder(tgrid,*minspos,mode='only close points',toll=1.5*grid_spacing,ignore_length_warning=True)
        detached_forw_ind=np.nonzero(difftgrid>1.5*grid_spacing)[0] #poits that have no immidiate grid point follower.
        detached_back_ind=np.nonzero(difftgrid>1.5*grid_spacing)[0]+1 #poits that have no immidiate grid point predecessor.
        det_for_use=detached_forw_ind[1:]  #we don't need the first forward and the last backward and will check them later
        det_back_use=detached_back_ind[:-1]  
        mask_in_det_interest=(det_for_use-det_back_use==1)*(point_to_inter[det_for_use]-point_to_inter[det_back_use]==0)   #close and equal: duplication of the same point 
        det_for_use_flag=det_for_use[mask_in_det_interest]   #we flag the forwards
        if len(detached_forw_ind)>0 and len(detached_back_ind)>0:
            if detached_forw_ind[0]==1 and point_to_inter[1]-point_to_inter[0]==0:  #==1 means it's the secondo point, so that there could be a duplicated min before it
                det_for_use_flag.insert(int(1),0)    #we flag the forwards
            
            if detached_back_ind[-1]<len(tgrid)-1: #it's not the last point: there could be a duplicated min after
                if point_to_inter[detached_back_ind[-1]+1]-point_to_inter[detached_back_ind[-1]]==0: #they are the same 
                    det_for_use_flag.append(detached_back_ind[-1]+1)   #and we flag the forward point
            
            if len(det_for_use_flag)>0:  #now we check wether there are poits to be popped out
                #Now we tile the index of the mins 2xlen(det_for_use_flag) and do the same we did in best index finder: we want to eliminate the furthest from the minimum 
                min_index_matrix=np.tile(index_of_mins_in_tgrid,(2*len(det_for_use_flag),1))  #the min index in tgrid is the best index to approximate the minimum in tgrid: the closest point of the duplication is the one with the same index in the grid
                tot_vec_flag=np.concatenate((det_for_use_flag-1,det_for_use_flag))   #back_flag-forw_flag; the back flag is just (for-flag-1)
                test_mat=min_index_matrix-np.reshape(tot_vec_flag,(len(tot_vec_flag),1)) 
                mask_mat=test_mat!=0  #true only where there is a non 0 element: evry 0 is a False
                non_zero_rows=mask_mat.all(axis=1)   #I take only the rows where there are no falses
                index_to_pop=tot_vec_flag[non_zero_rows]  
                tgrid=np.delete(tgrid,index_to_pop)   #we eliminate those from the grid
                point_to_inter=np.delete(point_to_inter,index_to_pop)
        
            
    return tgrid,point_to_inter
#Trace_filtering: those functions can be useful to apply a low pass bessel filter to a trace
#Filtering: Those functions deals with the optimal filtering issue
def absolute_grid_sum(trace,tvec,grid_spacing=0.5,return_grid=False):   #It's inspired to INterp_grid
    """
    Computes the absolute sum of windowed signal sums over a time grid.
    Parameters
    ----------
    trace : array_like
        Signal to analyze.
    tvec : array_like
        Time vector (uniformly sampled).
    grid_spacing : float, optional
        Grid spacing in seconds. Default = 0.5.
    return_grid : bool, optional
        If True, also returns the grid center times and per-cell contributions.
    Returns
    -------
    absolute_grid_difference : float
        Sum of the absolute values of the sums over the grid windows.
    tgrid, grid_diff : ndarray, optional
        Returned only if `return_grid=True`.
    """
    dts = np.diff(tvec)
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=1e-12):
        raise ValueError("tvec must be uniformally sampled")
    dt=dts.mean()    #This is the sample rate uniformity test we have used frequently
    int_step=int(grid_spacing/dt)   #We calculate the index step based on the idea that tvec(i+1)=tvec(i)+dt
    grid=np.arange(int_step,len(tvec),int_step,dtype=int)  #we build the grid index: we just need this
    grid_diff=np.zeros(len(grid))   #We preallocate the space
    if len(grid)==0:
        raise ValueError("empty grid")  #we avoid this situation
    else:
        for i in range(len(grid)):  #now we have to cycle on the grid index. We take the points around the i point of the grid. If P_i is the i-th point of the grud then we take all of the points in [P_(i-1),P_(i+1)]
            if len(grid)==1:                #if the grid is made of only one point
                maskinter=range(0,len(trace))      #we take all of the points in the trace
            else:
                if i==0:                                  #first point: P_(-1):=trace[0] clearly
                    maskinter=range(0,grid[i+1])
                elif i==len(grid)-1:                              #P_(len(grid)):=trace[-1]
                    maskinter=range(grid[i-1],len(trace))
                else:
                    maskinter=range(grid[i-1],grid[i+1])     #in most of the cases we simply apply the rule above 
            if len(trace[maskinter])>0:
                grid_diff[i]=np.sum(trace[maskinter])   #if in the interval there is at least one point we sum all the values
            else:
                grid_diff[i]=np.nan   #else we put the sum to nan
        
    mask=np.isnan(grid_diff)   #here we search for the nans
    grid_diff=grid_diff[~mask]  #we eliminate the points
    absolute_grid_difference=np.sum(np.abs(grid_diff))   #then we sum all of the points in absolute values
    if not return_grid:
        return absolute_grid_difference 
    else:
        tgrid=tvec[grid]   #if needed we extract the time grid
        tgrid=tgrid[~mask]   #and we eliminate and eliminate the nans
        return absolute_grid_difference,tgrid,grid_diff   #we can also return the grid 

