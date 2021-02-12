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


def correlation(exp_spec):
    """ Performs 2D correlation analysis.

    Calculates the Synchronous and Asynchronous 2D correlation of a given
    Spectrum.
    Hereby the dynamic spectrum is taken by subtracting the average.

    Parameters:
    ----------
    exp_spec : DataFrame
            Matrix of measured Spectrum

    Returns:
    -------
    sync_spec, async_spec: array_like
            The synchronous respectively asynchronous
            correlation spectrum
    """
    # calculating average and dynamic spectrum as numpy array
    index = list(exp_spec.index)
    exp_spec = exp_spec.to_numpy()
    dyn_spec = exp_spec - exp_spec.mean(axis=1)[:, None]

    # getting number of rows and columns
    rows = dyn_spec.shape[0]
    cols = dyn_spec.shape[1]

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
    # Work_Note: maybe reduce calculation due to symmetry?
    for i in range(rows):
        for k in range(rows):
            sync_spec[i, k] = np.sum(dyn_spec[i] * dyn_spec[k]) / (cols - 1)
            async_spec[i, k] = np.sum(dyn_spec[i]
                                      * np.sum(np.matmul(h_n_m, dyn_spec[k]))
                                      ) / (cols - 1)
    # alternative
    # for i in range(rows):
    #    for k in range(rows):
    #       sync_spec[i, k] = np.dot(dyn_spec[i, None], dyn_spec[k, None].T)/(
    #              cols - 1)

    # returns Spectra as DataFrame
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
