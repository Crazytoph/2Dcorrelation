"""Module for the Analysis.

Collection of useful functions to perform operations on Data.

Functions:
---------
    centering(arr, axis):
        centers data around axis
    pareto_scaling(arr,axis):
        scales data around axis
    correlation(*exp, ref):
        performs 2D correlation analysis on 'circ_data'
    max_wave(df, wave_min, wave_max):
        gives max. value and its wavelength for each column.
    min_wave(df, wave_min, wave_max):
        gives min. value and its wavelength for each column.
    interpolate(df, i):
        interpolates a function throuhg column points for index 'i'.
    derivative(df):
        gives the interpolated derivative of a DataFrame.
    sigmoid(x, a, b):
        a sigmoid function
    sigmoid_fit(df, wave):
        fit a sigmoid function on the data of 'df'
"""

# imports
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.misc
from scipy.optimize import curve_fit


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
    center: like dtype of arr
        centered data
    """
    # check for type 'np.ndarray', if not expect 'pd.DataFrame' and convert respectively
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
        center = pd.DataFrame(center, index=idx, columns=col)

    return center


def normalize(arr, axis=1):
    """Normalizes data between 0 and 1."""
    # check for type 'np.ndarray', if not expect 'pd.DataFrame' and convert respectively
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()

    # get array shape
    shape = arr.shape

    if axis == 1:
        min_col = np.array(arr.min(axis=axis))
        max_col = np.array(arr.max(axis=axis))
        diff = max_col - min_col
        diff = np.matmul(diff[:, None], np.ones((1, shape[1])))
        min_col = np.matmul(min_col[:, None], np.ones((1, shape[1])))
        norm_arr = (arr + abs(min_col)) / diff
    if axis == 0:
        min_idx = np.array(arr.min(axis=axis))
        max_idx = np.array(arr.min(axis=axis))
        diff = max_idx - min_idx
        diff = np.matmul(diff[None, :], np.ones((shape[0], 1)))
        min_idx = np.matmul(min_idx[None, :], np.ones((shape[0], 1)))
        norm_arr = (arr + abs(min_col)) / diff

    # reformat if necessary
    if df:
        norm_arr = pd.DataFrame(norm_arr, index=idx, columns=col)

    return norm_arr


def pareto_scaling(arr, axis=1):
    """Performs Pareto-scaling on data.

    Performs pareto-scaling according to [1]_ ,

    ..math: \tilde{x}_{ij} = \frac{x_{ij}-\bar{x_{i}}}{\sqrt{s_{i}}}

    Parameter:
    ---------
    arr: numpy array or dataframe
    axis: int
        can be 0 for column or 1 for row

    Return:
    ------
    pareto: like dtype of arr
        scaled date

    References:
    ----------
    ..[1] van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K., & van der Werf, M. J. (2006).
     Centering, scaling, and transformations: Improving the biological information content of metabolomics data.
     BMC Genomics, 7. https://doi.org/10.1186/1471-2164-7-142
    """
    # check for type 'np.ndarray', if not expect 'pd.DataFrame' and convert respectively
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()

    # perform pareto scaling
    avg = arr.mean(axis=axis)  # mean
    std = arr.std(axis=axis)  # standard deviation
    if axis == 0:
        pareto = (arr - np.reshape(avg, (1, len(avg)))) / np.sqrt(np.reshape(std, (1, len(std))))
    if axis == 1:
        pareto = (arr - np.reshape(avg, (len(avg), 1))) / np.sqrt(np.reshape(std, (len(std), 1)))

    # reformat if necessary
    if df:
        pareto = pd.DataFrame(pareto, index=idx, columns=col)

    return pareto


def correlation(*exp_spec, ref_spec=None, scaling=None):
    """ Performs 2D correlation analysis.

    Calculates the Synchronous and Asynchronous 2D correlation of a given
    Spectrum according to [1]_ .
    Hereby the dynamic spectrum is taken by subtracting the average.

    Parameters:
    ----------
    exp_spec: DataFrames
        Matrix of measured Spectrum, one or two. If one, homogenous
        DataFrame is calculated, if two heterogeneous.
    ref_spec: DataFrame
        Reference Spectrum, if None Average is taken to calculate Dynamic
        Spectrum.
    scaling: String
        Defines type of scaling if there is no reference spectrum, can be 'pareto' or 'centering'

    Returns:
    -------
    sync_spec, async_spec: array_like
        The synchronous respectively asynchronous
        correlation spectrum

    References:
    ----------
        ..[1]: Noda, I. (2000). Determination of Two - Dimensional Correlation Spectra Using the Hilbert Transform 5 E.
         54(7), 994–999.
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
        dyn2 = exp2 - ref_spec

    # perform scaling if wanted
    if scaling == 'pareto':
        dyn1 = pareto_scaling(exp1)
        dyn2 = pareto_scaling(exp2)

    # getting number of rows and columns
    rows = dyn1.shape[0]
    cols = dyn1.shape[1]

    # creating 2d arrays for sync and async spectra
    size = (rows, rows)
    sync_spec = np.zeros(size)
    async_spec = np.zeros(size)

    # creating Hilbert-Noda-matrix for async spectrum
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


def max_wave(df, wave_min=None, wave_max=None):
    """Returns the maximum value of its wavelength for each column.

    Parameter:
    ---------
    df : DataFrame
        Typical Wavelength Temperature DataFrame

    wave_min, wave_max: int
        max. and min. wavelength in between which the maximum should be found. Must be index values from the DataFrame.

    Return:
    ------
    df2 : DataFrame
        DataFrame with Temperatures as Index and under 'Value' column their max. CD value and under 'Wavelength' the
        corresponding wavelength.
    """
    # adapt to min. and max. wavelength
    if wave_min is not None:
        df = df.loc[wave_min:]
    if wave_max is not None:
        df = df.loc[:wave_max]

    # get max.value and max. wavelength
    max_val = pd.DataFrame(df.max(), columns=["Value"])
    max_wav = pd.DataFrame(df.idxmax(), columns=["Wavelength"])
    df2 = pd.DataFrame.join(max_val, max_wav)

    return df2


def min_wave(df, wave_min=None, wave_max=None):
    """Returns the minimum value and its wavelength for each column.

    Parameter:
    ---------
    df : DataFrame
        Typical Wavelength Temperature DataFrame

    wave_min, wave_max: int
        max. and min. wavelength in between which the minimum should be found. Must be index values from the DataFrame.

    Return:
    ------
    df2 : DataFrame
        DataFrame with Temperatures as Index and under 'Value' their smallest value and under 'Wavelength' the
        corresponding wavelength.
    """
    # adapt to min. and max. wavelength
    if wave_min is not None:
        df = df.loc[wave_min:]
    if wave_max is not None:
        df = df.loc[:wave_max]

    # get max.value and max. wavelength
    min_val = pd.DataFrame(df.min(), columns=["Value"])
    min_wav = pd.DataFrame(df.idxmin(), columns=["Wavelength"])
    df2 = pd.DataFrame.join(min_val, min_wav)

    return df2


def interpolate(df, i=0):
    """Interpolates a function for a given DataFrame 'df' for the row 'i'.

        Kind is 'cubic', out of data range points are extrapolated.

    Returns:
    -------
        f : function
            function which can be called to get interpolated value of value x.
    """
    x = list(df.columns)
    y = list(df.iloc[i, :])
    f = scipy.interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    return f


def derivative(df):
    """Tries to find the interpolated derivative of a DataFrame values a 'df'

    First a function is interpolated with 100 DataPoints between min. and max. column value
    which is then differentiated.

    Returns:
    -------
        deriv: DataFrame
            DataFrame with the 'Wavelength' as Index and the 100 points as columns.
    """
    # create x values
    x = np.linspace(df.columns[0], df.columns[-1], 100)
    deriv = pd.DataFrame(x, columns=['Wavelength'])

    # first interpolate function and then get derivative values
    for i in range(len(df.index)):
        f = interpolate(df, i)
        values = scipy.misc.derivative(f, x)
        deriv[i + 200] = values

    # reformat dataframe to original
    deriv.set_index('Wavelength')
    deriv = deriv.T

    return deriv


def sigmoid(x, a, b):
    """A sigmoid function.

    ..math: sig(t) = \frac{1}{1 + \exp{-a*(x-b)}

    Returns:
    -------
        function value on 'x'
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def sigmoid_deriv(x, a, b):
    """Calculates the derivative of func: sigmoid.

    Returns:
    -------
        derivative value  of f on 'x'
    """
    f = a * np.exp(-a * (x-b)) / ((1 + np.exp(-a * (x-b)))**2)
    return f


def sigmoid_fit(df, wave=247, a_range=[0, 0.3], b_range=[50, 70]):
    """Fits a sigmoid function on data on wavelength 'wave' in 'df'.

    Data must be normalized! b is somewhat the y=0.5 value, a the width of the function.

    Parameters:
    ----------
        df: DataFrame
            Data
        wave: int
            wavelength to be fitted
        a_range, b_range:  list with two integers
            min. and max. guessing value for parameters for optmial fit.

    Return:
    ------
        fit_data: DataFrames
            DataFrame with Temperatures as Columns and then index-wise: the Wavelength 'wave' and its orignal data,
            on 'fit' the fitted data and on 'up' and 'down' the fit with parameters shifted up/down about the standard
            deviation from the fit.
    """
    # get x and y points
    x_data = list(df.columns)
    y_data = df.loc[wave, :]  # get y points
    y_data.index = y_data.index.astype(float)

    # prepare DataFrames
    x = np.arange(df.columns[0], df.columns[-1] + 1, 1)
    fit_data = pd.DataFrame(x, index=x, columns=["wavelength"])
    fit_data = pd.concat([fit_data, y_data], axis=1)
    fit_data = fit_data.set_index("wavelength")

    # fit best parameters and their errors
    popt, pcov = curve_fit(sigmoid, x_data, y_data, method='trf',
                           bounds=([a_range[0], b_range[0]], [a_range[-1], b_range[-1]]))

    # get fit curve
    fit_data["fit"] = sigmoid(x, *popt)
    # get up and down error fit
    std = np.sqrt(np.diag(pcov))
    print(popt, std)
    fit_data["up"] = sigmoid(x, *(popt + std))
    fit_data["down"] = sigmoid(x, *(popt - std))

    return fit_data.T, popt
