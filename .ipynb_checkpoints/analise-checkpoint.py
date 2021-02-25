"""Module for the Analysis.

Collection of useful functions to perform operations on Data.

Functions:
---------
correlation(circ_data):
    performs 2D correlation analysis on 'circ_data'
maximum(arr):
    gets the maximum value of 'arr'

"""

# imports
import numpy as np
import pandas as pd  # to be deleted
import scipy.interpolate


def centering(arr, axis=1):
    """Centers data by subtracting the average.

    Centering given data by subtracting the average of respectively each
    column('axis'=0) or each row('axis'=1).

    Parameter:
    ---------
    arr: numpy array or dataframe
    axis: int
        can be 0 for column or 1 for row
    Return:
    ------
    center: like arr
        centered data
    """
    # check for type
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()
    # center
    avg = arr.mean(axis=axis)
    if axis == 0:
        center = arr - np.reshape(avg, (1, len(avg)))
    else:
        center = arr - np.reshape(avg, (len(avg), 1))

    # reformat if necessary
    if df:
        center = pd.DataFrame(arr, index=idx, columns=col)

    return center


def correlation(*exp_spec, ref_spec=None):
    """ Performs 2D correlation analysis.

    Calculates the Synchronous and Asynchronous 2D correlation of a given
    Spectrum.
    Hereby the dynamic spectrum is taken by subtracting the average.

    Parameters:
    ----------
    exp_spec: DataFrames
        Matrix of measured Spectrum, one or two. If one, homogenous
        DataFrame is calculated, if two heterogeneous.
    ref_spec: DataFrame
        Reference Spectrum, if None Average is taken to calculate Dynamic
        Spectrum.

    Returns:
    -------
    sync_spec, async_spec: array_like
        The synchronous respectively asynchronous
        correlation spectrum
    """
    # transform dataFrame to numpy
    index = list(exp_spec[0].index)
    exp1 = exp_spec[0].to_numpy()
    exp2 = exp_spec[-1].to_numpy()

    # create dynamic spectrum
    if ref_spec is None:
        dyn1 = centering(exp1)
        dyn2 = centering(exp2)
    else:
        ref_spec = ref_spec.to_numpy()
        dyn1 = exp1 - ref_spec
        dyn2 = exp2 - ref_spec  # maybe centering afterwards?

    # getting number of rows and columns
    rows = dyn1.shape[0]
    cols = dyn1.shape[1]

    # creating 2d arrays for sync and async spectra
    size = (rows, rows)
    sync_spec = np.zeros(size)
    async_spec = np.zeros(size)

    # creating Hilbert_Noda_matrix for async spectrum
    arr = np.arange(1, rows + 1)
    h_n_m = arr - np.array([arr]).T + np.identity(rows)
    h_n_m = 1 / (np.pi * h_n_m)
    h_n_m = (h_n_m - h_n_m.T) / 2
    h_n_m = h_n_m[..., :cols]

    # calculating sync and async values for each row and column
    for i in range(rows):
        for k in range(rows - 1):
            sync_spec[i, k] = np.sum(dyn1[i] * dyn2[k]) / (cols - 1)
            async_spec[i, k] = np.sum(dyn1[i]
                                      * np.sum(np.matmul(h_n_m, dyn2[k]))
                                      ) / (cols - 1)
    # alternative
    # for i in range(rows):
    #    for k in range(rows):
    #       sync_spec[i, k] = np.dot(dyn_spec[i, None], dyn_spec[k, None].T)/(
    #              cols - 1)

    # complete other half
    sync_spec = sync_spec + sync_spec.T - np.diag(sync_spec.diagonal())
    async_spec = async_spec + async_spec.T - np.diag(async_spec.diagonal())
    # return spectra as DataFrame
    sync_spec = pd.DataFrame(sync_spec, index=index, columns=index)
    async_spec = pd.DataFrame(async_spec, index=index, columns=index)
    return sync_spec, async_spec


def maximum(arr):
    """Returns the maximum value of array 'arr'."""
    return np.amax(arr)


def interpolate(df):
    """Interpolates a function for a given DataFrame. """
    x = list(df.columns)
    y = list(df.iloc[0, :])
    f = scipy.interpolate.interp1d(x, y, kind='cubic')
    return f
