"""Module for the hot plots.

Functions:
---------
heatmap_plot(df):
    plots heatmap of an dataframe
function_plot(df):
    plots values for selected indices over the column values

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def heatmap_plot(df, min=0, limit=None):
    """Plots heatmap.

    This function plots a heatmap of a DataFrame 'df' from 'min' to 'limit'
    rows and columns.

    Parameters:
    ----------
    df: DataFrame
        data for plot
    min, limit: int
        number of wavelength which should be skipped on the left side resp.
        max. number of wavelength

    Notes:
    -----
    To-Do:  defining map-style, adapt min, limit to nm-input,
    input change to array?

    """

    # getting max value
    if limit is None:
        max = len(df) + 1
    else:
        max = limit + 1

    # creating array from DataFrame, create Figure and plot
    arr = pd.DataFrame.to_numpy(df.iloc[min:max, min:max])
    fig, ax = plt.subplots()

    im = ax.imshow(arr, cmap='tab20c', vmax=abs(arr).max(), vmin=-abs(
        arr).max())

    # make it nice
    ax.set_title('Title')
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_xticks(np.arange(len(df.columns)), list(df.columns))
    ax.set_yticks(np.arange(len(df)), list(df.index))
    fig.colorbar(im)
    # Displaying the figure
    plt.show()
    plt.close(fig)


def function_plot(df, df2=None, idx=[0], min=0, limit=None):
    """Plots simple graph of selected idx

    This function plots several CD values for each selected Temperature Column
    'idx'

    Parameters:
    ----------
    df: DataFrame
        data for plot
    idx: list
        index which should be plotted
    min, limit: int
        number of wavelength which should be skipped on the left side resp.
        max. number of wavelength

    Notes:
    -----
    To-Do:  defining line-style, adapt min, limit to nm-input,
    input change to array?
    """
    # setting max
    if limit is None:
        max = len(df.columns) + 1
    else:
        max = limit + 1

    x = list(df.columns)[min:max]  # getting x-values
    fig, ax = plt.subplots()  # create figure containing single axis

    for i in idx:  # iterating through all selected colums
        y = pd.DataFrame.to_numpy(df.iloc[i, min: max])     # getting y-values
        ax.plot(x, y, 'gx--', linewidth=1, label=i)  # plot

    if df2 is not None:
        x2 = list(df2.columns)[min:]
        y2 = pd.DataFrame.to_numpy(df2.iloc[i, min:])
        ax.plot(x2, y2, 'r,--', linewidth=1, label='interpol')

    ax.set_xlabel('Temperature[K]')  # Add an x-label to the axes.
    ax.set_ylabel('CD [mdeg]')  # Add a y-label to the axes.
    ax.set_title("CD values")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()  # show
    plt.close(fig)
