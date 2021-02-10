"""Module for the hot plots.

Functions:
---------
heatmap(df):
    plots heatmap of an dataframe
function(df):
    plots values for selected indices over the column values

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def heatmap(*df, x_min=None, x_max=None, y_min=None, y_max=None,
            x_label="temperature[K]", y_label="wavelength[nm]", title="Heatmap"
            ):
    """Plots heatmap.

    This function plots heatmaps of DataFrames 'df' from in given area ('x_min,
    'x_max, 'y_min', y_max')

    Parameters:
    ----------
    *df: DataFrames
        data for plot
    x_min, x_max: int
        min and max value of index which should be plotted
    y_min, y_max: int
        min and max value of column which should be plotted
    title, x_label, y_label: string
        description for plot

    Notes:
    -----
    To-Do:  defining map-style, adapt min, limit to nm-input,
    input change to array?

    """
    # create running variable and figure
    k = 1
    fig = plt.figure()
    # get min, max values if not specified
    if x_min is None:
        x_min = list(df[0].index)[0]
    if x_max is None:
        x_max = list(df[0].index)[-1]
    if y_min is None:
        y_min = list(df[0].columns)[0]
    if y_max is None:
        y_max = list(df[0].columns)[-1]

    # for each dataframe, get plot
    for i in df:
        arr = pd.DataFrame.to_numpy(i.loc[x_min:x_max, y_min:y_max])
        arr = arr.T
        ax = fig.add_subplot(1, len(df), k)
        im = ax.imshow(arr, aspect='auto', cmap='gist_earth',
                       vmax=5, vmin=-8, interpolation='bicubic',
                       extent=[y_min, y_max, x_min, x_max], origin="lower")
        k = k + 1
        # make it nice
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im)

    # Displaying the figure
    plt.show()
    plt.close(fig)



def function(rows, *args, df2=None, x_min=None, x_max=None,
             x_label="Temperature[K]", y_label="CD values[mdeg]",
             title="Nice Plot"):
    """Plots simple graph of DataFrames

    This function plots several CD values for each selected Temperature Column
    'r'

    Parameters:
    ----------
    *args: Tuple of DataFrames
        data for plot
    df2: DataFrame
        data for 2nd subplot
    rows: list
        rows of DataFrames which should be plotted
    x_min, x_max: int
        min and max columns index which should be plotted
    title, x_label, y_label: string
        description for plot

    Notes:
    -----
    To-Do:  defining line-style, adapt min, limit to nm-input,
    input change to array?
    """
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    k, c = 0, 0
    fig = plt.figure()  # create figure

    # iterate through df in args and plot the rows
    ax = fig.add_subplot(1, 2, 1)
    for df in args:
        for i in rows:
            x = list(df.loc[:, x_min:x_max].columns)
            y = pd.DataFrame.to_numpy(df.loc[i, x_min:x_max])
            ax.plot(x, y, linestyle='--', color=colors[k], linewidth=1, label=i
                    )
            k = k + 1

    ax.set_xlabel(x_label)  # Add an x-label to the axes.
    ax.set_ylabel(y_label)  # Add a y-label to the axes.
    ax.set_title(title)  # Add a title to the axes.
    ax.legend()  # Add a legend.

    # if df2 exist, 2nd subplot is plotted
    if df2 is not None:
        for i in rows:
            ax2 = fig.add_subplot(1, 2, 2)
            x2 = list(df2.loc[:, x_min:x_max].columns)
            y2 = pd.DataFrame.to_numpy(df2.loc[i, x_min:x_max])
            ax2.plot(x2, y2, linestyle='-.', color=colors[c], linewidth=2,
                     label=i)
            c = c + 1

        ax2.set_xlabel(x_label)  # Add an x-label to the axes.
        ax2.set_ylabel(y_label)  # Add a y-label to the axes.
        ax2.set_title(title)  # Add a title to the axes.
        ax2.legend()  # Add a legend.

    # show plot and then close figure
    plt.show()
    plt.close(fig)
