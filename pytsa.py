#!/usr/bin/env python
# -*- coding: ascii -*-
#-----------------------------------------------------------------------------
"""
Time Series Analysis
pytsa (read "pizza") depends on scipy and numpy.

Pytsa is a simple timeseries utility for python.
It is good for pedagogical purposes, such as to understand moving averages,
linear regression, interpolation, and single/double/triple exponential smoothing.
I plan to add VAR, ARMA, etc.
"""

__author__ = 'Vedant Misra'
__license__ = 'BSD'
__vcs_id__ = '$Id$'
__version__ = '0.0.1' 
#Versioning: http://www.python.org/dev/peps/pep-0386/


# Initialization
import random
import numpy
import scipy
import math
import matplotlib.pyplot as plt
from scipy import stats


def mean(data):
    """
    Returns the average of the values in data.

    Parameters
    ----------
    yvals : ndarray
	The values of which we want to compute the mean.


    Returns
    -------
    mean : float
	The average of values in data
    """
    mean = numpy.mean(data)
    return mean


def errors(data, fits=None, method='diff'):
    """
    Returns an array of values that are the element-wise difference
    between the entries in fits and data.

    Parameters
    ----------
    data : ndarray
	The original values of which 'fits' is presumably a fit
    fits : ndarray, optional
	Values with which we want to compare the entires in 'data'
	If this isn't provided, the mean of the values in 'data'
	is used to calculate pairwise errors.
    method : string, optional
	'diff' : compute fits[i] - data[i]
	'sq' : compute the squares of elements returned by 'diff'
	'sumsq' : compute the sum of values returned by 'squared'
	'meansq' : compute the mean of values returned by 'squared'
    
    Returns
    -------
    errors : ndarray or float
	Return type is determined by the type of 'method'
	'diff' : ndarray
	'sq' : ndarray
	'sumsq' : float
	'meansq' : float
	
    """
    errs = numpy.array([])
    if fits == None:
	for i in range(len(data)):
	    fits = numpy.append(fits, mean(data))

    if method == 'diff':
        for i in range(len(data)):
	    errs = numpy.append(errs, (fits[i] - data[i]))
    elif method == 'sq':
	for i in range(len(data)):
	    errs = numpy.append(errs, (fits[i] - data[i]) ** 2.0)
    elif method == 'sumsq':
	sqerrs = errors(data, fits, 'sq')
	sum = 0
	for sqerr in sqerrs:
	    if math.isnan(sqerr) == False:
		sum += sqerr
	errs = sum
    elif method == 'meansq':
	ssqerrs = errors(data, fits, 'sumsq')
	n = 0
	for e in fits[:len(data)]:
	    if math.isnan(e) == False:
		n += 1
	errs = ssqerrs / n
    return errs


def sma(data, window):
    """
    Compute the simple moving average (sma) of data.
    
    The first [window - 1] entries are NaN.  For every other position t, the 
    value is the average of values at t, t-1, ..., t - window + 1.
    
    Parameters
    ----------
    data : ndarray
	The data for which we want to compute the sma.
    window : ndarray
	The size of the window for which we want to average values.

    Returns
    -------
    smas : ndarray
	The simple moving average of elements of data.
    
    """
    smas = numpy.array([])
    for i in range(window - 1):
	smas = numpy.append(smas, float('nan'))
    for i in range(window-1, len(data)):
	smas = numpy.append(smas, (mean(data[i - window + 1 : i + 1])))
    return smas


def cma(data, window):
    """
    Compute the centered moving average (cma) of data.
    
    The first [window - 1] entries are NaN.  For odd values of window, this is 
    the same as the simple moving average with values shifted up by 
    (window - 1)/2.  For even values of window, values are shifted up by 
    (window / 2) - (1/2) and averaged at adjacent positions.

    Parameters
    ----------
    data : ndarray
	The data for which we want to compute the cma.
    window : ndarray
	The size of the window for which we want to average values.

    Returns
    -------
    cmas : ndarray
	The centered moving average of elements of data.
    """
    cmas = numpy.array([])
    smas = sma(data, window)
    # if window is even
    if int(window / 2.0) == (window / 2.0):
	smasAvg = numpy.array([])
	for i in range(window, len(smas)):
	    next = (smas[i] + float(smas[i-1])) / 2.0
	    smasAvg = numpy.append(smasAvg, next)
    # if window is odd
    elif int(window / 2.0) != (window / 2.0):
	smasAvg = smas[window-1:]
    # build cmas
    for i in range(int(window / 2.0)):
	cmas = numpy.append(cmas, float('nan'))
    cmas = numpy.append(cmas, smasAvg)
    for i in range(int(window / 2.0)):
	cmas = numpy.append(cmas, float('nan'))
    return cmas


def linreg(tvals, data, fc = 0):
    """
    Compute the least-squares regression for data.

    Parameters
    ----------
    tvals : ndarray
	The independent variable values that correspond to each element of
	data.  Must be the same length as data.
    data : ndarray
	The data for which we want a linear regression.
    fc : int, optional
	Number of periods to forecast ahead.

    Returns
    -------
    res : tuple, comprised of:
	tvals_ext : ndarray
	    If fc = 0, tvals is the same as the input tvals.  Otherwise, it is
	    extended by fc periods.
	yvals : ndarray
	    Dependent-variable coordinates (ordinates) in the linear
	    regression.
	r: float
	    Pearson correlation coefficient.
	p: float
	    Two-sided p-value for a hypothesis test whose null hypothesis is 
	    that the slope is zero.

    """

    tvals_ext = tvals
    (a_s, b_s, r, p, std_err) = stats.linregress(tvals, data)
    if fc > 0:
	try:
	    diff = tvals[-1] - tvals[-2]
	except:
	    diff = 1
	for i in range(0, fc):
	    tvals_ext = numpy.append(tvals_ext, tvals_ext[-1] + diff)
    yvals = scipy.polyval([a_s, b_s], tvals_ext)
    return (tvals_ext, yvals, r, p)

def lerp(tvals, data, tvals2 = None, method='reconstruct'):
    """
    Perform linear interpolation (lerp) of two sets of measurements.

    Parameters
    ----------
    tvals : ndarray
	The independent variable values that correspond to each element of
	data. Must be the same length as data.
    data : ndarray
	The data we want to linearly interpolate.
    tvals2 : ndarray, optional
	A second set of (presumably evenly-spaced) independent variable
	points to which to interpolate.  If this isn't provided, it will
	be computed using whatever technique is specified in method.
    method : string, optional
	The method by which to generate an interpolated set of tvals.
	'reconstruct' : the smallest interval in tvals is added repeatedly
	    to the smallest value in tvals
	'pad' : the values of tvals are retained, but gaps are filled in.
	    This does NOT yield evenly-spaced tvals and should probably
	    be avoided.
	'fill' : len(tvals) remains the same, but the points are evenly
	    spaced in that interval

    Returns
    -------
    tvals_int : ndarray
	Interpolated values of the independent variable
    data_int : ndarray
	Interpolated data
    """
    
    diffs = []
    for i in range(1, len(tvals)):
	diffs.append(tvals[i] - tvals[i-1])
    interval = min(diffs)

    if tvals2 != None:
	tvals_int = tvals2
    if method == 'fill' and (tvals2 == None):
	tvals_int = numpy.linspace(min(xvals), max(xvals), len(xvals))
    elif method == 'reconstruct':
	tvals_int = numpy.array([])
	mn = min(tvals)
	mx = max(tvals)
	next = mn
	c = 0
	while add < mx:
	    next = mn + c * interval
	    tvals_int.append(add)
	    c += 1
    elif method == 'pad' and (tvals2 == None):
	tvals_int = numpy.array([])
	tvals_int = numpy.append(tvals_int, tvals[0])
	for i in range(1, len(tvals)):
	    diff = tvals[i] - tvals[i-1]
	    if diff == interval:
		tvals_int = numpy.append(tvals_int, tvals[i])
	    else:
		num = int(diff / interval)
		for c in range(1, num):
		    next = tvals[i-1] + c * interval
		    tvals_int = numpy.append(tvals_int, next)
		tvals_int = numpy.append(tvals_int, tvals[i])
    
    data_int = numpy.interp(tvals_int, tvals, data)
    return [tvals_int, data_int]



def singleES(tvals, data, a=None, fc = 0):
    """
    Smoothe a timeseries using single exponential smoothing.  Does not
    recognize trends or periodicity.

    Parameters
    ----------
    tvals : ndarray
	The independent variable values that correspond to each element of
	data. Must be the same length as data.
    data : ndarray
	The data we want to smoothe.
    a : float, optional
	The alpha coefficient in single exponential smoothing.
	If a isn't given, it is computed using the Levenberg-Marquardt
	least squares minimization algorithm.
    fc : int, optional
	The number of periods to forecast ahead.

    Returns
    -------
    ret : tuple, comprised of:
	tvals_int : ndarray
	    Equal to tvals if fc = 0. If fc > 0, is an extension of tvals.
	smoothed : ndarray
	    Smoothed data
    """
    # Subfunction that computes errors for single exponential smoothing
    def singleESError(a, tvals, data):
	fits = singleES(tvals, data, a)[1]
	err = errors(data, fits, 'meansq')
	return err
    if a == None:
	a = scipy.optimize.leastsq(singleESError, 0.5, 
	    args=(tvals, data))[0]
	if a > 1: a = 1
	if a < 0: a = 0
	#print "alpha:", float(a)
    a = float(a)
    interval = tvals[1] - tvals[0]
    smoothed = numpy.array([float('nan'), data[0]])
    for i in range(2, len(tvals)):
	next = a * data[i-1] + (1-a)*(smoothed[-1])
	smoothed = numpy.append(smoothed, next)
    fc = int(fc)
    if fc == 1:
	next = a * data[-1]
	tvals = numpy.append(tvals, tvals[-1] + interval)
	smoothed = numpy.append(smoothed, next)
    elif fc > 1:
	boot = data[-1]
	for i in range(0, fc):
	    tvals = numpy.append(tvals, tvals[-1] + interval)
	    next = a * boot + (1 - a)*(smoothed[-1])
	    smoothed = numpy.append(smoothed, next)
    ret = (tvals, smoothed)
    return ret

def doubleES(tvals, data, params=None, init=None, fc = 0):
    """
    Smoothe a timeseries using double exponential smoothing. Recognizes trends
    but not periodicity.

    Parameters
    ----------
    tvals : ndarray
	The independent variable values that correspond to each element of
	data. Must be the same length as data.
    data : ndarray
	The data we want to smoothe.
    params : list, optional
	The parameters alpha and beta, or alpha and gamma, in a list.  That is,
	[alpha, beta].  If either alpha or beta is not provided, the parameters
	that minimize the MSE are computed using the Levenberg-Marquardt
	algorithm.
    init : None, int, float, or 'full'; optional
	Initialization method for the intermediary (trend) time series.
	Behavior varies depending on type(init).
	None : trend time series initialized with average of first two elements
	    of data
	int : trend time series initialized with average of first init elements
	    of data
	float : trend time series initialized with init
	'full' : trend time series initialized with the quotient of the 
	    difference between the last and first elements of data, and the
	    number of periods, less one.
    fc : int, optional
	Number of periods to forecast ahead.

    Returns
    -------
    ret : tuple, comprised of
	tvals : ndarray
	    If fc = 0, this is the same as the input tvals.  If fc != 0, 
	    it is tvals extended to include additional periods.
	smoothed : ndarray
	    The smoothed values.
    """
    # Subfunction that computes errors for double exponential smoothing
    def doubleESError(params, tvals, data, init=None):
	fits = doubleES(tvals, data, params, init)[1]
	err = errors(data, fits, 'meansq')
	return [err, err]
    if params == None or (None in params):
	params0 = [0.5, 0.5]
	params = scipy.optimize.leastsq(doubleESError, params0,
	    args=(tvals, data, init))[0]
	for val in params:
	    if val > 1: val = 1
	    if val < 0: val = 0
	print "alpha:", params[0]
	print "beta:", params[1]
    interval = tvals[1] - tvals[0]
    # initialize bvals
    if init == None:
	bvals0 = (data[1] - data[0]) / 2.0
    elif type(init) == type(int(1)):
	bvals0 = mean(data[:init + 1])
    elif type(init) == type(float(1)):
	bvals0 = init
    elif init == 'full':
	bvals0 = (data[-1] - data[0]) / (len(data) - 1)
    bvals = numpy.array([bvals0])
    smoothed = numpy.array([data[0]])
    a = params[0]
    b = params[1]
    for i in range(1, len(tvals)):
	nextS = (a * data[i]) + ((1-a) * (smoothed[-1] + bvals[-1]))
	nextB = (b * (nextS - smoothed[-1])) + ((1-b) * bvals[-1])
	smoothed = numpy.append(smoothed, nextS)
	bvals = numpy.append(bvals, nextB)
    fc = int(fc)
    if fc >= 1:
	bootS = smoothed[-1]
	bootB = bvals[-1]
	for i in range(0, fc):
	    tvals = numpy.append(tvals, tvals[-1] + interval)
	    nextS = bootS + i * bootB
	    smoothed = numpy.append(smoothed, nextS)
    ret = (tvals, smoothed)
    return ret

def ar():
    pass

def tripleExponentialSmoothing(a, b, c, xvals, yvals, fc=0):
    [xint, yint] = linearInterpolation(xvals, yvals, method='reconstruct')
    diff = xint[1] - xint[0]
    if True:
	inter1 = (yint[3] - yint[0]) / 3.0
    elif False:
	inter1 = yint[1] - yint[0]

    return data

def test():
    """ 
    Tests module components.
    """
    def genData(n, min, max):
	data = numpy.array([])
	while numpy.size(data) < n:
	    data = numpy.append(data, random.randint(min, max))
    #tvals = numpy.array(range(1, 11))
    #data = numpy.array([6.4, 5.6, 7.8, 8.8, 11.0, 11.6, 16.7, 15.3, 21.6, 22.4])
    tvals = range(1, 31)
    data = genData(30, 0, 100)
    data = numpy.array(data)
    (tvals1, fits1) = singleES(tvals, data, fc=5)
    (tvals2, fits2) = doubleES(tvals, data, fc=5)
    (tvals3, fits3) = linReg(tvals, data, fc=5)
    printLists(
	[tvals, data, errors(data), errors(data, method='sq'),
	sma(data, 4), cma(data, 4), fits1, fits2, fits3]
	['time', 'data', 'errors', 'squares', 'sma', 'cma', 'ses', 'des', 
	    'linreg'])
    print "Mean:", mean(data)
    print "SSE:", errors(data, method='sumsq')
    print "MSE Mean:", errors(data, method='meansq')
    print "MSE Single:", errors(data, fits1, 'meansq')
    print "MSE Double:", errors(data, fits2, 'meansq')
    
    printLists([tvals, data, fits1, fits2],['time', 'data', 'ses', 'des'])

    plt.plot(xvals, yvals, 'k+', 
	     xint, fits, 'b', 
	     xint2, fits2, 'r', 
	     xint2p, fits2p, 'g')
    plt.show()


def printLists(lists, headers=None):
    if headers != None:
	for header in headers:
	    print header,
	    print " " * (14 - len(header)),
	print ""
    longest = max([len(l) for l in lists])
    for i in range(longest):
	for list in lists:
	    try:
		if math.isnan(list[i]):
		    print "  -----  " + "\t",
		else:
		    print "%f" % list[i] + "\t",
	    except:
		print "  -----  " + "\t",
	print ""
	
	
if __name__=='__main__':
    test()
