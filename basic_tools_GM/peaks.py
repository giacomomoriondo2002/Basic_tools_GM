import numpy as np
from .utils import best_index_finder
__all__ = [
    "extr_finder",
    "rel_peaks_finder",
    "add_section",
    "relevant_peaks_mask",
]
def extr_finder(trace):
    """
    This function finds local extrema (maxima and minima) in a one-dimensional signal.
    Parameters
    ----------
    trace : array_like
        One-dimensional signal to analyze.
    Returns
    -------
    extr_ind : list of int
        Indices of all critical points (local maxima or minima).
    max_ind : list of int
        Indices of local maxima.
    min_ind : list of int
        Indices of local minima.
    Notes
    -----
    - An extremum is detected when the sign of the first-difference derivative changes.
    - Plateaus (flat regions) may be classified as local minima.
    - Extrema at the signal boundaries are not considered.
    """
    dtrace = np.diff(trace)
    #OLD procedure. It's not optimal because of the cyclings required.
    # zero_ind = []
    # extr_ind=[]
    # # Trova i punti dove il segno della derivata cambia (massimi o minimi)
    # for i in range(len(dtrace) - 1):
    #     if dtrace[i] * dtrace[i + 1] <= 0:
    #         zero_ind.append(int(i))
    #         extr_ind.append(int(i+1))
    # max_ind = []
    # min_ind=[]
    # for ind in zero_ind:
    #     # Verifica se Ã¨ un massimo locale
    #     if trace[ind] <= trace[ind + 1] >= trace[ind + 2]:
    #         max_ind.append(int(ind + 1))
    #     else:
    #         min_ind.append(int(ind+1))
    #Optimixed procedure: no cycle and just vectorisation
    diffshiftup=dtrace[1:]
    diffshiftdown=dtrace[:-1]
    prodvec=diffshiftup*diffshiftdown  #the relevant index is the down one
    zero_ind=np.array(np.nonzero(prodvec<=0),dtype=int)[0]  #index in prodvec where there is a 0==>we have to shift it since diff loses an index
    extr_ind=zero_ind+1
    #Now we identify which point is a maximum ad which a minum 
    points=trace[extr_ind]   #the extremants found
    pointsup=trace[extr_ind+1]  #points immediatly after the extremants
    pointsdown=trace[extr_ind-1]  #points immediatly before the extremats
    
    testup=points-pointsup     
    testdown=points-pointsdown
    maskmaxup=testup>=0
    maskmaxdown=testdown>=0
    totmask=maskmaxup*maskmaxdown   #In this mask the True value are given only to the maxima points
    max_ind=extr_ind[totmask]
    min_ind=extr_ind[~totmask]
   
    
    return extr_ind, max_ind, min_ind
#This function is made to find the relevant peaks
def rel_peaks_finder(trace,delta,toll=0,return_extremant_indiced=False):
    """
    Finds "relevant" local peaks in a one-dimensional signal, based on a minimum 
    amplitude difference (`delta`) compared to neighboring extrema.
    Parameters
    ----------
    trace : array_like
        One-dimensional signal to analyze.
    delta : float
        Minimum amplitude difference a peak must exceed with respect to its adjacent 
        minima (or the signal boundaries) to be considered relevant.
    toll : int, optional, default=0
        Tolerance for including nearby secondary peaks that do not fully meet the 
        `delta` criterion. If within `toll` extrema of a valid peak, they are 
        considered relevant as well.
    return_extremant_indiced : bool, optional, default=False
        If True, also returns all extrema indices (maxima and minima) detected 
        by `extr_finder`.
    Returns
    -------
    relevant : list of int
        Indices of relevant local peaks.
    extr_ind : list of int, optional
        Indices of all extrema (maxima and minima) in the signal. Returned only if 
        `return_extremant_indiced=True`.
    maxind : list of int, optional
        Indices of local maxima in the signal. Returned only if 
        `return_extremant_indiced=True`.
    minind : list of int, optional
        Indices of local minima in the signal. Returned only if 
        `return_extremant_indiced=True`.
    Notes
    -----
    - The function uses `extr_finder` to detect local maxima and minima.
    - A peak is considered relevant if it exceeds `delta` compared to both 
    neighboring minima (or compared to signal edges for the first/last extrema).
    - The `toll` parameter allows capturing secondary peaks that are close to 
    relevant ones and exceed the threshold on at least one side.
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 0, 2, 0, 1, 0])
    >>> relevant, extr_ind, maxind, minind = relevant_peaks_finder(
    ...     x, delta=0.5, return_extremant_indiced=True
    ... )
    >>> relevant
    [1, 3, 5]
    """
    relevant=np.array([])    #we define an empty array to be filled with the index of the relevant peaks
    extr_ind,maxind,minind=extr_finder(trace) #we extract the indices of the extremant
    
    if toll==0: #this is the only section needed
        minuse=np.asarray(minind,dtype=int)
        if extr_ind[0]==maxind[0]: #first is a max
            minuse=np.insert(minuse,0,0)
        if extr_ind[-1]==maxind[-1]: #last is a max
            minuse=np.insert(minuse,len(minuse),len(trace)-1)
        mindown=minuse[:-1]
        minup=minuse[1:]
        upcond=trace[maxind]-trace[minup]>delta
        downcond=trace[maxind]-trace[mindown]>delta
        centralcond=upcond*downcond
        relevant=maxind[centralcond]
    if toll>0:  #in this case we use this approach
        minuse=np.asarray(minind,dtype=int)
        if extr_ind[0]==maxind[0]: #first is a max
            minuse=np.insert(minuse,0,0)
        if extr_ind[-1]==maxind[-1]: #last is a max
            minuse=np.insert(minuse,len(minuse),len(trace)-1)
        mindown=minuse[:-1]
        minup=minuse[1:]
        upcond=trace[maxind]-trace[minup]>delta
        downcond=trace[maxind]-trace[mindown]>delta
        centralcond=upcond*downcond    #the central ones
        relevant=maxind[centralcond]
        for max in maxind[downcond]:  #we take only the the maxima respecting the down condition
            if max not in relevant: #if already taken we don't need to check this peak
                ind_in_maxind=np.nonzero(maxind==max)[0][0]  #index of max in maxind: since max is in maxind we don't need a robust code and this is faster
                ind_to_sum=np.arange(1,min(len(maxind),toll+1))  #we sum up to this: if we can go up to toll we go; otherwise we stop at the last maximum
                summed_indices=ind_to_sum+ind_in_maxind #the maxima we need to check
                mask=np.isin(maxind[summed_indices],maxind[upcond])  #we check whether those respects the up condition(already evaluated)
                if any(mask): #if we have at least one true
                    ind_of_first_true=np.argmax(mask)  #index of first True
                    ind_for_appending=ind_of_first_true+1 #if it's 0, then we have to append the one after max and so on
                    peaks_to_append=maxind[ind_in_maxind:ind_in_maxind+ind_for_appending+1]
                    relevant=np.concatenate((relevant,peaks_to_append))
    relevant=np.sort(relevant)   
    if return_extremant_indiced:
        return relevant,extr_ind,maxind,minind
    else:
        return relevant
#This funcion also identifies a boolean mask containing the relevant peaks point. Those are the points going from the minimu before to the one after a relevant peak.
#Auxiliar function for the main function
def add_section(start,stop,mask):
    for i in range(start,stop):
        if 0<i<len(mask)-1:
            mask[i]=True
def relevant_peaks_mask(trace,delta,toll=0,include_mins=True,return_mins=False):  
    """
    Identifies relevant peaks in a signal and builds a boolean mask
    highlighting the significant regions around those peaks.
    Parameters
    ----------
    trace : array_like
        One-dimensional signal to analyze.
    delta : float
        Minimum amplitude threshold to consider a peak as relevant
        (difference between local maximum and local minimum).
    toll : int, optional
        Maximum number of consecutive peaks to tolerate when they do not fully
        meet the threshold criterion. Default = 0.
    include_mins : bool, optional
        If True, include minima in the mask. Default = True.
    return_mins : bool, optional
        If True, include minima in the output (for optimization efficiency). Default = False.
    Returns
    -------
    relevant : list of int
        Indices of the relevant peaks detected in the signal.
    mask : ndarray of bool
        Boolean mask with the same length as `trace`, True over sections considered
        significant (around relevant peaks) and False elsewhere.
    Notes
    -----
    - The function uses `extr_finder` to obtain the local extrema of the signal.
    - Sections marked as True in the mask are obtained by expanding the region
    around the selected peaks:
    * Central expansion: between the previous and the next minimum.
    * Forward/backward expansion: as long as the amplitude between peaks exceeds `delta`.
    """
    if len(trace) < 3:   #In this case we prefer not to interrupt the program.
        return [], np.zeros_like(trace, dtype=bool)
    mask=np.zeros(len(trace),dtype=bool)  #we pre-allocate a all False vector
    relevant,extr_ind,_,minind=rel_peaks_finder(trace,delta,toll=toll,return_extremant_indiced=True)  #we find the relevant peaks
    relevant_set = set(relevant)   #faster for routine check
    #We use a cycle here because it's very hard to vectorialise this process without complicating excessively the code
    for i, h in enumerate(extr_ind):
        if h in relevant_set:   #we only work with the relevant peaks
            ind_h_in_rel=np.nonzero(relevant==h)[0][0] #index of h in relevant. since h is in relevant we don't need a robust code: this is faster
            #central expansion
            if i>0 and i<len(extr_ind)-1:  #case h is not the fist extrament or the last one: those case must be treated separatly
                startc=extr_ind[i-1]   #the minimum before
                stopc=extr_ind[i+1]    #the minimum after
            elif i==0:   #first extremant
                startc=int(0)    #we take trace[0] as minimum before
                stopc=extr_ind[i+1]
            else :    #last extrament
                startc=extr_ind[i-1]
                stopc=len(trace)-1   #we use trace[-1] as last minimum
            add_section(startc,stopc+1,mask)   #we add the sections
            #forward expansion
            j=i+2  #the maximum after h 
            if ind_h_in_rel+1<len(relevant)-1:    #if it's not the last relevant index
                stopfor=relevant[ind_h_in_rel+1]   #this says that we will stop if we get to the next relevant index
            elif i<len(extr_ind)-1:  #if it's not the last extremant but the last relevant
                stopfor=minind[-1]   #if it's the last relevant peak we will stop if we get to the last minimum 
            while j+1<len(extr_ind)-1 and  extr_ind[j]<stopfor:   #this checks that the cycling is possible (we have at least a maximumm afetr h) and that, if present, they resepct the condition above
                if trace[extr_ind[j]]-trace[extr_ind[j+1]]>delta:   #if the j th peak respects the forward condition (peak-right_min>delta)
                    add_section(extr_ind[j-1],extr_ind[j+1]+1,mask)  #we add the points to the relevant set
                    j+=2  #we check the following peak
                else:   #if the forward condition is not respected
                    break  #we stop
            #backward expansion
            j=i-2  #this is the maximum before h
            if ind_h_in_rel-1>=0:   #if it's not the first relevant peak
                stopback=relevant[ind_h_in_rel-1]  #we will stop there
            else:
                stopback=minind[0]   #we sopt at the first minimum
            while j>=1 and extr_ind[j]>stopback:    #if is at least 1 (so we can use j=0 if the last estremant is a minimum)
                if trace[extr_ind[j]]-trace[extr_ind[j-1]]>delta:  #if we have the back condition
                    add_section(extr_ind[j-1],extr_ind[j+1]+1,mask)  #we add the points
                    j-=2    #and check the minima before
                else:
                    break  #if it's not true we stop
    if not include_mins:
        #Old code:
        #mask[minind]=False   #The problem with this was that it is faster, but also includes minima we don't want 
        #New code: fixing for undesired minima and vectorialised: it's faster
        # I have to consider the minima, but I want the minima not to be on the rising or descending side of a great maximum
        #We start by searching for the minima indices in the extr_ind vectors
        ind_of_mins_in_extr_ind_raw=best_index_finder(extr_ind,*minind) #we must use best_index since it can deal with multi target problems
        ind_of_mins_in_extr_ind_center=np.copy(ind_of_mins_in_extr_ind_raw)
        #now to check this we must consider the max before and after a minimum, so if we start and stop with a minimum we must check them separatly
        if ind_of_mins_in_extr_ind_raw[0]==0: #if tra trace starts with a minimum
            back_cond_first=trace[0]-trace[extr_ind[0]]>delta
            for_con_first=trace[extr_ind[1]]-trace[extr_ind[0]]>delta
            ind_of_mins_in_extr_ind_center=ind_of_mins_in_extr_ind_raw[1:]   
        if ind_of_mins_in_extr_ind_raw[-1]==len(extr_ind)-1: #if ends end with a minimum
            for_cond_last=trace[-1]-trace[extr_ind[-1]]>delta
            back_cond_last=trace[extr_ind[-2]]-trace[extr_ind[-1]]>delta
            ind_of_mins_in_extr_ind_center=ind_of_mins_in_extr_ind_center[:-1]   
        #now the maximum before is the index-1
        ind_of_maxs_bef_in_extr_ind_center=ind_of_mins_in_extr_ind_center-1
        #the maximum afetr is the index+1
        ind_of_maxs_af_in_extr_ind_center=ind_of_mins_in_extr_ind_center+1
        #conditions for(wards) and back(wards)
        mask_in_ind_of_mins_back=trace[extr_ind[ind_of_maxs_bef_in_extr_ind_center]]-trace[extr_ind[ind_of_mins_in_extr_ind_center]]>delta  #the maximum before is higher then the minimum under exam with greater than a delta differnce
        mask_in_ind_of_mins_for=trace[extr_ind[ind_of_maxs_af_in_extr_ind_center]]-trace[extr_ind[ind_of_mins_in_extr_ind_center]]>delta    #the maximum after is higher then the minimum under exam with greater than a delta differnce
        if ind_of_mins_in_extr_ind_raw[0]==0: #if starts with a minimum
            mask_in_ind_of_mins_for=np.insert(mask_in_ind_of_mins_for,0,for_con_first)
            mask_in_ind_of_mins_back=np.insert(mask_in_ind_of_mins_back,0,back_cond_first)
        if ind_of_mins_in_extr_ind_raw[-1]==len(extr_ind)-1: #if ends end with a minimum
            mask_in_ind_of_mins_for=np.insert(mask_in_ind_of_mins_for,len(mask_in_ind_of_mins_for),for_cond_last)
            mask_in_ind_of_mins_back=np.insert(mask_in_ind_of_mins_back,len(mask_in_ind_of_mins_back),back_cond_last)
        #NOW we have to check because there may be some cases to keep a minimum. We built the negative mask
        #left checking: decsending part of a maximum
        #We have to check the last minimum separatly
        mask_in_ind_of_mins_back_shift=mask_in_ind_of_mins_back[1:]   #to go left we need the back cond
        if ind_of_mins_in_extr_ind_raw[-1]==len(extr_ind)-1: #if ends end with a minimum we can not say anithing so the condition is False
            mask_in_ind_of_mins_back_shift=np.insert(mask_in_ind_of_mins_back_shift,len(mask_in_ind_of_mins_back_shift),False)
        else:   #if it ends with a max
            back_cond_last_point=trace[extr_ind[-1]]-trace[-1]>delta   #we see if the last point respect the back condition.
            mask_in_ind_of_mins_back_shift=np.insert(mask_in_ind_of_mins_back_shift,len(mask_in_ind_of_mins_back_shift),back_cond_last_point)
        mask_in_ind_of_mins_left=mask_in_ind_of_mins_back*mask_in_ind_of_mins_back_shift*(~mask_in_ind_of_mins_for)  #if the min under analysis has the back, the min after has the back, and the minimum ander analysis hasn't the forwad: it's True
        #right checking: same but on the right; it's the rising part of a relevant peak
        mask_in_ind_of_mins_for_shift=mask_in_ind_of_mins_for[:-1]   #to go right we need the for cond
        if ind_of_mins_in_extr_ind_raw[0]==0: #if it starts with a minimum we can not say anithing so the condition is False
            mask_in_ind_of_mins_for_shift=np.insert(mask_in_ind_of_mins_for_shift,0,False)
        else:   #if it starts with a max
            for_cond_last_point=trace[extr_ind[0]]-trace[0]>delta   #we see if the first point respect the for condition.
            mask_in_ind_of_mins_for_shift=np.insert(mask_in_ind_of_mins_for_shift,0,for_cond_last_point)
        mask_in_ind_of_mins_right=mask_in_ind_of_mins_for*mask_in_ind_of_mins_for_shift*(~mask_in_ind_of_mins_back)   #if the min under analysis has the for, the min after has the for, and the minimum ander analysis hasn't the baxk: it's True
        mask_in_ind_of_mins_oscill=mask_in_ind_of_mins_right + mask_in_ind_of_mins_left  #acts like an or
        selected_mins=extr_ind[ind_of_mins_in_extr_ind_raw[~mask_in_ind_of_mins_oscill]]   #We need the all-min vector
        mask[selected_mins]=False
    if return_mins:
        return relevant,mask,minind
    else:
        return relevant,mask
#This function is useful to interpolate signals 

