"""Module for the Analysis.

Collection of useful functions to perform operations on Data.

Functions:
---------
    centering(arr, axis):
        centers data around axis
    normalize(arr, axis=1):
        normalize data from 0 to 1
    projection_matrix(data_df, rows, alpha=0, positive_projection=True):
        creates mixture of projection and residue matrix of certain rows
    auto_scaling(arr, axis=1):
        scales data by the standard deviation along axis
    pareto_scaling(arr,axis):
        scales data by the root of the standard deviation along axis
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
import lmfit


def centering(df):
    """Centers data by subtracting the average.

    Centering given data by subtracting the average of respectively each
    column.

    Parameter:
    ---------
    df: Dataframe

    Return:
    ------
    center: like dtype of arr
        centered data
    """
    # center
    avg = weighted_mean(df)
    center = df - avg

    return center


def weighted_mean(df):
    """ Calculates the weighted mean of unevenly-spaced data.

    Based on:
    Noda, I. and Y.Ozaki.
    “Two - Dimensional Correlation Spectroscopy: Applications in Vibrational and Optical Spectroscopy.” (2002).
    Chapter: "Practical Computation of 2D Correlation Spectra"

     Parameter:
        ---------
        df: original DataFrame

        Return:
        ------
        mean: numpy array
            vector with mean value per row
    """
    # get col, idx and convert to numpy
    col = df.columns
    idx = df.index
    arr = df.to_numpy()

    # get Temperature as List ad T_0 and T_(M+1)
    temp_list = list(col)
    temp_list = [2 * temp_list[0] - temp_list[1]] + temp_list + [temp_list[-1] * 2 - temp_list[-2]]

    # calculate mean
    norm = 0
    mean = np.zeros((idx.size, 1))
    for i in range(col.size):
        mean = mean + arr[:, i, None] * (temp_list[i + 2] - temp_list[i])
        norm = norm + (temp_list[i + 2] - temp_list[i])
    mean = mean / norm
    return mean


def weighted_std(df):
    """ Calculates the weighted mean of unevenly-spaced data.

        Based on:
        Noda, I. and Y.Ozaki.
        “Two - Dimensional Correlation Spectroscopy: Applications in Vibrational and Optical Spectroscopy.” (2002).
        Chapter: "Practical Computation of 2D Correlation Spectra"

        and "https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance" (11.05.2021)

         Parameter:
            ---------
            df: original DataFrame

            Return:
            ------
            std: numpy array
                vector with standard deviation of each row
        """
    # get col, idx and convert to numpy
    col = df.columns
    idx = df.index
    mean = weighted_mean(df)
    arr = df.to_numpy()

    # get Temperature as List ad T_0 and T_(M+1)
    temp_list = list(col)
    temp_list = [2 * temp_list[0] - temp_list[1]] + temp_list + [temp_list[-1] * 2 - temp_list[-2]]

    # calculate std
    norm = 0
    std = np.zeros((idx.size, 1))
    for i in range(col.size):
        std = std + (arr[:, i, None] - mean) ** 2 * (temp_list[i + 2] - temp_list[i])
        norm = norm + (temp_list[i + 2] - temp_list[i])

    return std / norm


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
        norm_arr = (arr - min_col) / diff
    if axis == 0:
        min_idx = np.array(arr.min(axis=axis))
        max_idx = np.array(arr.min(axis=axis))
        diff = max_idx - min_idx
        diff = np.matmul(diff[None, :], np.ones((shape[0], 1)))
        min_idx = np.matmul(min_idx[None, :], np.ones((shape[0], 1)))
        norm_arr = (arr - min_idx) / diff

    # reformat if necessary
    if df:
        norm_arr = pd.DataFrame(norm_arr, index=idx, columns=col)

    return norm_arr


def vector_norm(arr, axis=1):
    """Performs vector normalization."""
    # check for type 'np.ndarray', if not expect 'pd.DataFrame' and convert respectively
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()

    # get norm
    norm = np.linalg.norm(arr, axis=axis)

    if axis == 1:
        norm_arr = arr/norm.reshape(-1, 1)
    else:
        norm_arr = arr/norm

    # reformat if necessary
    if df:
        norm_arr = pd.DataFrame(norm_arr, index=idx, columns=col)

    return norm_arr


def auto_scaling(arr, axis=1):
    """Performs Auto-scaling on data, also called Pearson scaling.
    Performs pareto-scaling according to [1]_ ,
    ..math: \tilde{x}_{ij} = \frac{x_{ij}}{s_{i}}
    Parameter:
    ---------
    df: DataFrame
    Return:
    ------
    auto: like dtype of arr
        scaled data
        
    References:
    ----------
    ..[1] van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K., & van der Werf, M. J. (2006).
     Centering, scaling, and transformations: Improving the biological information content of metabolomics data.
     BMC Genomics, 7. https://doi.org/10.1186/1471-2164-7-142
    """
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()

    # perform pareto scaling
    avg = arr.mean(axis=axis)  # mean
    std = arr.std(axis=axis, ddof=1)  # standard deviation
    if axis == 0:
        auto = (arr - np.reshape(avg, (1, len(avg)))) / np.reshape(std, (1, len(std)))
    if axis == 1:
        auto = (arr - np.reshape(avg, (len(avg), 1))) / np.reshape(std, (len(std), 1))

    # reformat if necessary
    if df:
        auto = pd.DataFrame(auto, index=idx, columns=col)
        
    return auto


def pareto_scaling(arr, axis=1):
    """Performs Pareto-scaling on data.

    Performs pareto-scaling according to [1]_ ,

    ..math: \tilde{x}_{ij} = \frac{x_{ij}}{\sqrt{s_{i}}}

    Parameter:
    ---------
    df: DataFrame

    Return:
    ------
    pareto: like dtype of arr
        scaled data

    References:
    ----------
    ..[1] van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K., & van der Werf, M. J. (2006).
     Centering, scaling, and transformations: Improving the biological information content of metabolomics data.
     BMC Genomics, 7. https://doi.org/10.1186/1471-2164-7-142
    """
    df = False
    if not isinstance(arr, np.ndarray):
        df = True
        col = arr.columns
        idx = arr.index
        arr = arr.to_numpy()

    # perform pareto scaling
    avg = arr.mean(axis=axis)  # mean
    std = arr.std(axis=axis, ddof=1)  # standard deviation
    if axis == 0:
        pareto = (arr - np.reshape(avg, (1, len(avg)))) / np.sqrt(np.reshape(std, (1, len(std))))
    if axis == 1:
        pareto = (arr - np.reshape(avg, (len(avg), 1))) / np.sqrt(np.reshape(std, (len(std), 1)))

    # reformat if necessary
    if df:
        pareto = pd.DataFrame(pareto, index=idx, columns=col)

    return pareto


def projection_matrix(data_df, row_min_max, positive_projection_trick=True, center=True):
    """Returns a new data matrix with the projected portion of the 'idx'-rows relative to 'alpha'.
     Method is based on [1]_
     Parameters:
     ----------
        data_df: DataFrame
            original data
        row_min_max: list of min and max row
            rows to be used for projection vector
        positive_projection: boolean
            parameter determining whether a positive projection trick should be done
        center: boolean
            whether dynamic spectrum must be created

    References:
    ----------
        ..[1]: Noda, I. (2010). Projection two-dimensional correlation analysis. Journal of Molecular Structure,
         974(1–3), 116–126. https://doi.org/10.1016/j.molstruc.2009.11.047
    Return:
    ------
        null_space_proj: DataFrame
            null_space_projection matrix
     """
    # centering if wanted, arr
    if center:
        df = centering(data_df)
    else:
        df = data_df
    arr = df.T.to_numpy()
    row_len = 1     # number of rows for reshaping vector, default = 1

    # positive projection trick only for vector
    if row_min_max[0] != row_min_max[-1]:
        positive_projection_trick = False
        row_len = row_min_max[-1] - row_min_max[0] + 1

    # define projecting vector
    proj_vec = df.loc[row_min_max[0]:row_min_max[-1]].to_numpy().reshape(df.shape[1], row_len)

    # calculate projecting matrix as y (yT,y)^-1 yT A
    proj_mat = np.matmul(proj_vec,
                         np.matmul(
                             np.linalg.inv(np.matmul(proj_vec.T, proj_vec)),
                             np.matmul(proj_vec.T, arr)
                         )
                         )

    # positive projection trick: each vector for yT a_j < 0 is set zero
    if positive_projection_trick:
        for i in range(proj_mat.shape[1]):
            if np.matmul(proj_vec.T, proj_mat[:, i]) < 0:
                proj_mat[:, i] = np.zeros(proj_mat.shape[0])

    # calculate the null space projection by subtracting the projecting matrix from the dynamic spectrum
    null_space_proj = arr - proj_mat
    # return new spectrum as DataFrame
    null_space_proj = pd.DataFrame(null_space_proj.T, index=df.index, columns=df.columns)
    return null_space_proj


def perturbation_moving_window(data_df, window_size=3):
    """Calculates synchronous and asynchronous perturbation-correlation moving-window
    correlation spectra. 
    
    Parameters:
    ----------
        data_df: DataFrame
            original Data
        window_size: int
            window_size, must be uneven!
    
    Return:
    ------
        sync, assync: DataFrames
            Synchronous and Assynchronous Spectra

    Reference:
    ---------
        Noda, I. (2018). Advances in Two-Dimensional Correlation Spectroscopy (2DCOS).
        In Frontiers and Advances in Molecular Spectroscopy. Elsevier Inc.
        https://doi.org/10.1016/B978-0-12-811220-5.00002-2
    """
    # define basic variables
    temp_list = np.array(data_df.columns)
    rows, cols = data_df.shape
    m = int((window_size - 1) / 2)

    # creating template matrices for synchronous and asynchronous spectra
    sync = np.ones((rows, cols - 2 * m))
    assync = np.ones((rows, cols - 2 * m))

    # creating Hilbert-Noda-matrix for async spectrum
    arr = np.arange(1, window_size + 1)
    h_n_m = arr - np.array([arr]).T + np.identity(window_size)
    h_n_m = 1 / (np.pi * h_n_m)
    h_n_m = (h_n_m - h_n_m.T) / 2
    h_n_m = h_n_m[..., :window_size]

    # calculating spectra for each window
    for j in range(m, len(temp_list) - m):
        # dynamic temperature
        temp_mean = np.sum(temp_list[j - m:j + m + 1]) / window_size
        temp_dyn = temp_list[j - m:j + m + 1] - temp_mean
        # dynamic spectrum
        window_vector = data_df.loc[:, temp_list[j - m:j + m + 1]]
        dyn_spec = centering(window_vector).to_numpy()
        # calculating synchronous and asynchronous spectra in matrix form
        sync[:, j - m] = np.matmul(dyn_spec, temp_dyn) / (2 * m)
        assync[:, j - m] = np.matmul(dyn_spec, np.matmul(h_n_m, temp_dyn)) / (2 * m)

    # transforming to DataFrame
    sync = pd.DataFrame(sync, index=data_df.index, columns=temp_list[m:-m])
    assync = pd.DataFrame(assync, index=data_df.index, columns=temp_list[m:-m])

    return sync, assync


def correlation(*exp_spec, ref_spec=None, center=False, scaling=None):
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
    center: Boolean
        whether centering should be done ADDITIONALLY after subtracting a reference spectrum,
        only relevant if one is given. Default: False
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
    # getting index and shared column values
    idx = exp_spec[0].index
    col = pd.concat(exp_spec, join='inner').columns

    # get single spectra
    exp1 = exp_spec[0]
    exp2 = exp_spec[-1]

    # create dynamic spectrum from average or reference or none if ref_spec == [0]
    if ref_spec is None:
        # from average spectrum
        dyn1 = centering(exp1.loc[:, col])
        dyn2 = centering(exp2.loc[:, col])
    elif isinstance(ref_spec[0], pd.core.frame.DataFrame):
        dyn1 = exp1.loc[:, col]
        dyn2 = exp2.loc[:, col]
    else:
        # from reference spectrum
        dyn1 = exp1.loc[:, col].subtract(ref_spec[0], axis=0)
        dyn2 = exp2.loc[:, col].subtract(ref_spec[-1], axis=0)

        # extra centering if wanted
        if center is True:
            dyn1 = centering(dyn1)
            dyn2 = centering(dyn2)

    # perform scaling if wanted
    if scaling == 'pareto':
        dyn1 = pareto_scaling(dyn1)
        dyn2 = pareto_scaling(dyn2)
    if scaling == 'auto':
        dyn1 = auto_scaling(dyn1)
        dyn2 = auto_scaling(dyn2)

    # transform to numpy array
    dyn1 = dyn1.to_numpy()
    dyn2 = dyn2.to_numpy()

    # get column and row numbers, create disrel_spec
    rows, cols = dyn1.shape
    disrel_spec = np.zeros((rows, rows))

    # creating Hilbert-Noda-matrix for async spectrum
    arr = np.arange(1, cols + 1)
    h_n_m = arr - np.array([arr]).T + np.identity(cols)
    h_n_m = 1 / (np.pi * h_n_m)
    h_n_m = (h_n_m - h_n_m.T) / 2
    h_n_m = h_n_m[..., :cols]
    
    # calculate synchronous and asynchronous spectrum with matrix
    sync_spec = np.dot(dyn1, dyn2.T) / (cols - 1)
    async_spec = np.dot(dyn1, np.dot(h_n_m, dyn2.T)) / (cols - 1)

    # disrelation spectra
    #for i in range(rows):
       # for j in range(rows):
           # kappa = np.sign(np.sum(dyn1[i, :-1] * dyn2[j, 1:] - dyn1[i, 1:] * dyn2[j, :-1]))
            #disrel_spec[i, j] = kappa * np.sqrt(sync_spec[i, i] * sync_spec[j, j] - sync_spec[i, j] * sync_spec[j,i])

    # return spectra as DataFrame
    sync_spec = pd.DataFrame(sync_spec, index=idx, columns=idx, dtype=float)
    async_spec = pd.DataFrame(async_spec, index=idx, columns=idx, dtype=float)
    #disrel_spec = pd.DataFrame(disrel_spec, index=idx, columns=idx, dtype=float)

    return sync_spec, async_spec, #disrel_spec


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


# general sigmoid fit and derivativ equal to s1
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
    f = a * np.exp(-a * (x - b)) / ((1 + np.exp(-a * (x - b))) ** 2)
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
    popt, pcov = curve_fit(sigmoid, x_data, y_data, method='dogbox',
                           bounds=([a_range[0], b_range[0]], [a_range[-1], b_range[-1]]))

    # get fit curve
    fit_data["fit"] = sigmoid(x, *popt)
    # get up and down error fit
    std = np.sqrt(np.diag(pcov))
    print(popt, std)
    fit_data["up"] = sigmoid(x, *(popt + std))
    fit_data["down"] = sigmoid(x, *(popt - std))

    return fit_data.T, popt, std


# functions based on origin documentation
# https://www.originlab.com/doc/Origin-Help/Curve-Fitting-Function#Growth.2FSigmoidal

def s1(x, a=1, y0=0, k=0.3, xc=63):
    """Sigmoidal Logistic function, type 1.

    Parameter:
    ---------
        x: float
            variable
        a: float
            amplitude
        xc: float
            center of sigmoidal, xc >0
        k: float
            coefficient

    Return:
    ------
        f: arry of floats
            fuction value on 'x'
    """
    f = (a / (1 + np.exp(-k * (x - xc)))) + y0
    return f


def linear(x, m=1, y0=0):
    """linear function"""
    f = m * x + y0
    return f


def lm_fit(df, wave=260, guess=[], f_type='s1', method='leastsq'):
    """Test function for trying out lm-fit module."""

    # get data to be fitted
    x = list(df.columns)
    data = df.loc[wave, :]  # get y points
    data.index = data.index.astype(float)

    # get model chosen by 'function'
    if f_type == 's1':
        model = lmfit.Model(s1)
    if f_type == 'linear':
        model = lmfit.Model(linear)
    if f_type == 'sigmoid':
        model = lmfit.Model(sigmoid)

    # change guess values of parameter
    for i in range(len(guess)):
        params = model.param_names
        model.set_param_hint(params[i], value=guess[i])

    # fit and print fit report
    result = model.fit(data, x=x, method=method, nan_policy='propagate')
    print(result.fit_report())

    # get confidence report and stuff
    # result.conf_interval()
    # print(result.ci_report())

    # prepare DataFrame for return
    fit_data = pd.DataFrame(x, index=x, columns=["wavelength"])
    fit_data = pd.concat([fit_data, data], axis=1)
    fit_data = fit_data.set_index("wavelength")
    fit_data["fit"] = result.best_fit
    fit_data["error"] = result.eval_uncertainty()
    fit_data["residual"] = result.residual

    # get stderr of params as dictionary
    stderr = {}
    for x in result.params.keys():
        stderr[x] = result.params[x].stderr

    return fit_data, result.params.valuesdict(), stderr
