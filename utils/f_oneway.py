import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def _square_of_sums(a, axis=0):
    """
    Sum elements of the input array, and return the square(s) of that sum.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.
    Returns
    -------
    square_of_sums : float or ndarray
        The square of the sum over `axis`.
    See also
    --------
    _sum_of_squares : The sum of squares (the opposite of `square_of_sums`).
    """
    a, axis = _chk_asarray(a, axis)
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s


def _sum_of_squares(a, axis=0):
    """
    Square each element of the input array, and return the sum(s) of that.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.
    Returns
    -------
    sum_of_squares : ndarray
        The sum along the given axis for (a**2).
    See also
    --------
    _square_of_sums : The square(s) of the sum(s) (the opposite of
    `_sum_of_squares`).
    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)


def f_oneway(args_):
    args = [np.asarray(arg, dtype=float) for arg in args_]
    # ANOVA on N groups, each in its own array
    num_groups = len(args)
    alldata = np.concatenate(args)
    bign = len(alldata)

    # Determine the mean of the data, and subtract that from all inputs to a
    # variance (via sum_of_sq / sq_of_sum) calculation.  Variance is invariance
    # to a shift in location, and centering all data around zero vastly
    # improves numerical stability.
    offset = alldata.mean()
    alldata -= offset

    sstot = _sum_of_squares(alldata) - (_square_of_sums(alldata) / bign)
    ssbn = 0
    for a in args:
        ssbn += _square_of_sums(a - offset) / len(a)

    # Naming: variables ending in bn/b are for "between treatments", wn/w are
    # for "within treatments"
    ssbn -= _square_of_sums(alldata) / bign
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / dfbn
    msw = sswn / dfwn
    f = msb / msw

    prob = stats.f.sf(dfbn, dfwn, f)   # equivalent to stats.f.sf
    #print('dfbn, dfwn, f: ', dfbn, dfwn, f)

    return f, prob
