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


def heatmap_plot(df):
    """Plots heatmap"""

    # Displaying dataframe as an heatmap
    # with diverging colourmap as RdYlBu
    plt.imshow(df, cmap="RdYlBu")

    # Displaying a color bar to understand
    # which color represents which range of data
    plt.colorbar()

    # Assigning labels of x-axis
    # according to dataframe
    plt.xticks(range(len(df)), df.columns)

    # Assigning labels of y-axis
    # according to dataframe
    plt.yticks(range(len(df)), df.index)

    # Displaying the figure
    plt.show()


def function_plot(df, cols, min=0, limit=None):
    """Plots simple graph of selected cols

    This function plots several CD values for each selected Temperature Column
    'cols'

    Parameters:
    ----------
    df: DataFrame
        data for plot
    cols: list
        columns which should be plotted
    min, limit: int
        number of wavelength which should be skipped on the left side resp.
        max. number of wavelength

    """
    # setting max
    if limit is None:
        max = len(df)
    else:
        max = limit

    x = list(df.index)[min:max] # getting x-values
    fig, ax = plt.subplots()  # create figure containing single axis

    for i in cols:  # iterating through all selected colums
        y = pd.DataFrame.to_numpy(df[i])    # getting y-values
        y = y[min:max]
        ax.plot(x, y, label=i)  # plot

    ax.set_xlabel('Wavelength[nm]')  # Add an x-label to the axes.
    ax.set_ylabel('CD [mdeg]')  # Add a y-label to the axes.
    ax.set_title("CD values")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()  # show
