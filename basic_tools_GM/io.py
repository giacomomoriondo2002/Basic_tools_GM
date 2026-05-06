import numpy as np
from datetime import datetime
from .utils import couple_sorting



__all__ = ["txt_file_writer"]





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
    auto_ordering : bool, optional
        If True, write arrays in decreasing order of length. If False, preserve
        the order given in ``names_of_arrays``.

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

