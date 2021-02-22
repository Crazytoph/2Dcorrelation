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
import ipywidgets as widgets


def heatmap(*df, x_min=None, x_max=None, y_min=None, y_max=None, swap=True,
            c_min=None, c_max=None, x_label="temperature[K]",
            y_label="wavelength[nm]", title="Heatmap"):
    """Plots heatmap.

       This function plots heatmaps of DataFrames 'df' from in given area ('x_min,
       'x_max, 'y_min', y_max')

       Parameters:
       ----------
       *df: DataFrames
           data for plot
       x_min, x_max: int
           Default: None
           min and max value of index which should be plotted
       y_min, y_max: int
           Default: None
           min and max value of column which should be plotted
       swap: boolean
           Default: True
           whether the axes should be swapped
       c_min, c_max: int
           min. and max. value for colorscale, if None, max and min value from
           array are taken
       title, x_label, y_label: string
           description for plot

       Notes:
       -----
       To-Do:  defining map-style, vmax, vmin?!
       """
    # define all basic parameters
    if x_min is None:
        x_min = df[0].index[0]
    if x_max is None:
        x_max = df[0].index[-1]
    if y_min is None:
        y_min = df[0].columns[0]
    if y_max is None:
        y_max = df[0].columns[-1]
    extent = [y_min, y_max, x_min, x_max]

    # ensure  we are interactive mode
    # this is default but if this notebook is executed out of order it may have been turned off
    plt.ion()
    fig = plt.figure()

    # set running variable 'k' to one and plot for each heatmap in df
    k = 1
    for i in df:
        ax = fig.add_subplot(1, len(df), k)
        k = k + 1
        arr = pd.DataFrame.to_numpy(i.loc[x_min:x_max, y_min:y_max])

        # if axis should be swapped
        if swap:
            arr = arr.T
            extent = [x_min, x_max, y_min, y_max]
            x_label, y_label = y_label, x_label
        if c_min is None:
            c_min = arr.min()
        if c_max is None:
            c_max = arr.max()

        # here happens the 'real' plot with all settings
        im = ax.imshow(arr, aspect='auto', cmap='gist_earth',
                       vmax=c_max, vmin=c_min, interpolation='bicubic',
                       extent=extent, origin="lower")

        # make it nice
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # make it nicer
    plt.title(title)
    plt.colorbar(im)

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def function(rows, *df, x_min=None, x_max=None, swap=False,
             x_label="Temperature[K]", y_label="CD values[mdeg]",
             ):
    """Plots simple graph of DataFrames

    This function plots several CD values for each selected Temperature Column
    'r'

    Parameters:
    ----------
    *df: Tuple of DataFrames
        data for plot
    rows: list
        rows of DataFrames which should be plotted
    x_min, x_max: int
        min and max columns index which should be plotted
    swap: boolean
        whether to change axes
    x_label, y_label: string
        description for plot

    Notes:
    -----
    To-Do:  defining line-style, adapt min, limit to nm-input,
    input change to array?
    """
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    k, c  = 0, 1
    plt.ion()
    fig = plt.figure()  # create figure
    # iterate through df in args and plot the rows
    
    for i in df:
        ax = fig.add_subplot(1, len(df), c)  # create subplot
        if swap:
            i = i.T
            x_min, x_max, y_min, y_max = y_min, y_max, x_min, x_max
        for r in rows:
            x = list(i.loc[:, x_min:x_max].columns)  # get x-values
            y = pd.DataFrame.to_numpy(i.loc[r, x_min:x_max])  # get y-values
            ax.plot(x, y, linestyle='-', marker=' ', color=colors[k],
                    linewidth=1, label=i
                    )
            k = k + 1
            ax.legend(rows)
        k = 0
        # plots line into graph
        ax.plot([20, 90], [0, 0], color='k', linestyle=':', linewidth=1)
        ax.set_xlabel(x_label)  # Add an x-label to the axes.
        ax.set_ylabel(y_label)  # Add a y-label to the axes.
        title = "fig." + str(c)
        c = c + 1
        plt.title(title)
        
def functionT(rows, *df, x_min = None, x_max = None, y_min = None, y_max = None, swap = True,
             x_label="Wavelength[nm]", y_label="CD values[mdeg]", y_scaling = None, baseline = None, line1 = None, line2 = None, line3 = None, line4 = None, line5 = None
             ):
    
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    k, c  = 0, 1
    plt.ion()
    fig = plt.figure()  # create figure
    # iterate through df in args and plot the rows
    
    for i in df:
        ax = fig.add_subplot(1, len(df), c)  # create subplot
        if swap:
            i = i.T
        # get min, max values if not given
        if x_min is None:
            x_min = i.columns[0]
        if x_max is None:
            x_max = i.columns[-1]
        if y_min is None:
            y_min = i.min()
        if y_max is None:
            y_max = i.max()
        
        for r in rows:
            x = list(i.loc[:, x_min:x_max].columns)  # get x-values
            y = pd.DataFrame.to_numpy(i.loc[r, x_min:x_max])  # get y-values
            ax.plot(x, y, linestyle='-', marker=' ', color=colors[k],
                    linewidth=1, label=i
                    )
            k = k + 1
            ax.legend(rows)
        k = 0
        
        if y_scaling is not None:         
            ax.set_ylim(y_scaling) # Y-achis scaling
        #ax.set_ylim([y_min,y_max]) # Y-achis auto-scaling
        if baseline is True: 
            ax.plot([210, x_max], [0, 0], color='k', linestyle='-', linewidth=1) # plots baseline into graph
        if line1 is not None:
            ax.plot([210, x_max], line1, color='k', linestyle=':', linewidth=1) # plots line into graph
        if line2 is not None:
            ax.plot([210, x_max], line2, color='k', linestyle=':', linewidth=1)
        if line3 is not None:
            ax.plot([210, x_max], line3, color='k', linestyle=':', linewidth=1)
        if line4 is not None:
            ax.plot([210, x_max], line4, color='k', linestyle=':', linewidth=1)
        if line5 is not None:
            ax.plot([210, x_max], line5, color='k', linestyle=':', linewidth=1)
        ax.set_xlabel(x_label)  # Add an x-label to the axes.
        ax.set_ylabel(y_label)  # Add a y-label to the axes.
        title = "Fig. " + str(c)
        c = c + 1
        plt.title(title)   
    
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )
