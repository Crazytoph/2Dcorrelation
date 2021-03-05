"""Module for the hot plots.

Functions:
---------
    heatmap(*df, x_min=None, x_max=None, y_min=None, y_max=None, swap=True,
            c_min=None, c_max=None, x_label="temperature[K]",
            y_label="wavelength[nm]", title="Heatmap"):
        Plots heatmap of an dataframe.
    function(rows, *df, x_min=None, x_max=None, swap=False,
             x_label="Temperature[K]", y_label="CD values[mdeg]", y_scaling=None
             ):
        Plots values for selected indices over the column values.
    functionT(rows, *df, x_min=None, x_max=None, y_min=None, y_max=None, swap=True,
              x_label="Wavelength[nm]", y_label="CD values[mdeg]", y_scaling=None,
              baseline=None, line1=None, line2=None, line3=None, line4=None, line5=None
              ):
        Plots values for selected indices over the column values specifically for Temperature
        on x-axis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from string import ascii_lowercase


def heatmap(*data, x_min=[], x_max=[], y_min=[], y_max=[], swap=True,
            c_min=[], c_max=[], x_label="temperature[K]",
            y_label="wavelength[nm]", title="Heatmap", subtitle=[]
            ):
    """Plots heatmap.
       This function plots heatmaps of DataFrames 'df' from in given area ('x_min,
       'x_max, 'y_min', y_max')
       Parameters:
       ----------
        *data: DataFrames or CData-object
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
        subtitle: list of strings
            titles for all subfigures
       Notes:
       -----
       To-Do:  defining map-style
       """
    # ensure  we are interactive mode
    # this is default but if this notebook is executed out of order it may have been turned off
    plt.ion()
    fig = plt.figure(facecolor="lemonchiffon")

    # set control variable variable 'k' to one and plot for each heatmap in df
    c = 1

    # plot each dataframe in a new subplot
    for i in data:
        # check the type
        if not isinstance(i, pd.core.frame.DataFrame):
            df = i.cd_df
        else:
            df = i

        # define all basic parameters
        if len(x_min) < c:
            x_min.append(df.index[0])
        if len(x_max) < c:
            x_max.append(df.index[-1])
        if len(y_min) < c:
            y_min.append(df.columns[0])
        if len(y_max) < c:
            y_max.append(df.columns[-1])
        extent = [y_min[c - 1], y_max[c - 1], x_min[c - 1], x_max[c - 1]]

        # create subplot and array
        if len(data) < 5:
            width = len(data)
        else:
            width = 4
        ax = fig.add_subplot((len(data) // 5) + 1, width, c)
        arr = pd.DataFrame.to_numpy(df.loc[x_min[c-1]:x_max[c-1], y_min[c-1]:y_max[c-1]])

        # if axis should be swapped
        if swap:
            arr = arr.T
            extent = extent[2:4] + extent[0:2]
            x_label, y_label = y_label, x_label

        # get colorscale range if not given
        if len(c_min) < c:
            c_min.append(arr.min())
        if len(c_max) < c:
            c_max.append(arr.max())

        # here happens the 'real' plot with all settings
        im = ax.imshow(arr, aspect='auto', cmap='gist_heat',
                       vmax=c_max[c-1], vmin=c_min[c-1], interpolation='bicubic',
                       extent=extent, origin="lower")

        # make it nice
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # add subtitle
        if len(subtitle) < c:
            ax.set_title("Fig. " + ascii_lowercase[c - 1])
        else:
            ax.set_title(subtitle[c - 1])

        # add coolorbar for each subfigur
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax2)
        c = c + 1  # push control variable

    # Title option
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # clear lists
    y_min.clear(), y_max.clear(), x_min.clear(), x_max.clear(), c_min.clear(), c_max.clear()

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def function(rows, *df, x_min=None, x_max=None, y_min=None, y_max=None, swap=False,
             x_label="Temperature[K]", y_label="CD values[mdeg]", title="", subtitle=[],
             y_scaling=False, baseline=False
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
        x_min, x_max, y_min, y_max: int
            min and max columns index which should be plotted
        swap: boolean
            whether to change axes
        x_label, y_label,titel: string
            description for plot
        y_scaling: list with two values
            gives min. and max. point on y-axis
        subtitle: list of strings
            title for each figure
    Returns:
    -------
        None  but plots.
    Notes:
    -----
    To-Do:  defining line-style,
    """
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

    c = 1  # control variable

    plt.ion()  # needed for jupyter
    fig = plt.figure()  # create figure

    # iterate through df in args and plot the rows
    for i in df:
        ax = fig.add_subplot(1,len(df), c)  # create subplot

        if swap:
            i = i.T

        # get min, max values if not given
        if x_min is None:
            x_min = i.columns[0]
        if x_max is None:
            x_max = i.columns[-1]
        if y_min is None:
            y_min = min(i.loc[rows].min())
        if y_max is None:
            y_max = max(i.loc[rows].max())

        k = 0  # control variable
        for r in rows:
            x = list(i.loc[:, x_min:x_max].columns)  # get x-values
            y = pd.DataFrame.to_numpy(i.loc[r, x_min:x_max])  # get y-values
            ax.plot(x, y, linestyle='-', marker='x', color=colors[k % 10],
                    linewidth=1
                    )
            k = k + 1

        # scaling
        if y_scaling is True:
            ax.set_ylim([y_min, y_max])  # Y-axis scaling
        # baseline
        if baseline is True:
            # plots line into graph
            ax.plot([x_min, x_max], [0, 0], color='k', linestyle=':', linewidth=1)

        # plots line into graph
        if baseline is not None:
            ax.plot([20, 90], [0, 0], color='k', linestyle='-', linewidth=1)

        # make it nice
        ax.set_xlabel(x_label)  # Add an x-label to the axes.
        ax.set_ylabel(y_label)  # Add a y-label to the axes.

        ax.legend(rows)  # add legend

        # add subtitle
        if len(subtitle) <= (c - 1):
            if len(df) > 1:
                ax.set_title("Fig. " + ascii_lowercase[c - 1])
        else:
            ax.set_title(subtitle[c - 1], fontsize=10)
        c = c + 1
        ax.set_title(title)

    # Title option
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(w_pad=0.5)

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def mult_func(rows, *probes, x_min=None, x_max=None, y_min=[], y_max=[], swap=False,
              x_label="Temperature[K]", y_label="CD values[mdeg]", title="", subtitle=[],
              marker=[], linestyle=[], label=None,
              y_scaling=False, baseline=False
              ):
    """Plots simple graph of DataFrames

    This function plots several CD values for each selected Temperature Column
    'r'

    Parameters:
    ----------
        *probes: Tuple or List of cdata.CData objects
            data for plot
        rows: list
            rows of DataFrames which should be plotted
        x_min, x_max, y_min, y_max: int
            min and max columns index which should be plotted
        swap: boolean
            whether to change axes
        x_label, y_label,titel: string
            description for plot
        y_scaling: list with two values
            gives min. and max. point on y-axis
        subtitle: list of strings
            title for each figure

    Returns:
    -------
        None  but plots.

    Notes:
    -----
    To-Do:  defining line-style,
    """
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

    # define function style for this plot

    c = 1  # control variable

    plt.ion()  # needed for jupyter
    fig = plt.figure()  # create figure

    # iterate through all lists for each subfigure
    for subs in probes:
        # create subplot and array
        if len(probes) < 5:
            width = len(probes)
        else:
            width = 4

        ax = fig.add_subplot((len(probes) // width)+1, width, c)  # create subplot
        # iterate through all data to b plotted into one subfigure
        for i in subs:
            # check instance
            if not isinstance(i, pd.core.frame.DataFrame):
                df = i.cd_df
            else:
                df = i

            # swap if wished, normally False
            if swap:
                df = df.T

             # get min, max values if not given
            if x_min is None:
                x_min = df.columns[0]
            if x_max is None:
                x_max = df.columns[-1]

            k = 0  # control variable
            # plot the data for all wanted rows
            for r in rows:

                # check whether r is in df
                if r not in df.index:
                    continue

                # define function style for this plot
                if len(marker) <= k:
                    marker.append('x')
                if len(linestyle) <= k:
                    linestyle.append('-')

                x = list(df.loc[:, x_min:x_max].columns)  # get x-values
                y = pd.DataFrame.to_numpy(df.loc[r, x_min:x_max])  # get y-values
                ax.plot(x, y, linestyle=linestyle[k], marker=marker[k],
                        color=colors[k % 10],
                        linewidth=1.5
                        )
                k = k + 1

        # scaling
        if y_scaling is True:
            ax.set_ylim([y_min, y_max])  # Y-axis scaling

        # baseline
        if baseline is True:
            ax.plot([x_min, x_max], [0, 0], color='k', linestyle=':', linewidth=1)

        # make it nice
        ax.set_xlabel(x_label)  # Add an x-label to the axes.
        ax.set_ylabel(y_label)  # Add a y-label to the axes.

        if label is None:  # add legend
            label = rows
        ax.legend(label)

        # add subtitle
        if len(subtitle) < (c - 1):
            ax.set_title("Fig. " + ascii_lowercase[c - 1])
        else:
            ax.set_title(subtitle[c - 1], fontsize=5)
        c = c + 1

    # Title option
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def functionT(rows, *df, x_min=None, x_max=None, y_min=None, y_max=None, swap=True,
              x_label="Wavelength[nm]", y_label="CD values[mdeg]", title="", subtitle=[],
              y_scaling=None, baseline=None,
              line1=None, line2=None, line3=None, line4=None, line5=None
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
            x_min, x_max, y_max, y_min: int
                min and max columns index which should be plotted
            swap: boolean
                whether to change axes
            x_label, y_label, title: string
                description for plot
                subtitle: lost of strings
            list of subfigure titles
            y_scaling: list with two values
                gives min. and max. point on y-axis
            baseline, line 1-5: boolean?
                activates help lines

        Returns:
        -------
            None  but plots.

        Notes:
        -----
        To-Do:  defining line-style,
        """
    # create color list and color variables
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

    c = 1  # control variable for later
    plt.ion()  # switch on interactive plot for jupyter
    fig = plt.figure()  # create figure

    # plot each DataFrame in new subplot
    for i in df:
        ax = fig.add_subplot(1, len(df), c)  # create subplot

        # swap for changing axis
        if swap:
            i = i.T

        # get min, max values if not given
        if x_min is None:
            x_min = i.columns[0]
        if x_max is None:
            x_max = i.columns[-1]
        if y_min is None:
            y_min = min(i.loc[rows].min())
        if y_max is None:
            y_max = max(i.loc[rows].max())

        # plot each 'r' row as new label with new color
        k = 0  # control variable
        for r in rows:
            x = list(i.loc[:, x_min:x_max].columns)  # get x-values
            y = pd.DataFrame.to_numpy(i.loc[r, x_min:x_max])  # get y-values
            ax.plot(x, y, linestyle='-', marker=' ', color=colors[k % 10],
                    linewidth=1, label=i
                    )
            k = k + 1

        #  set y-axis limits if wanted
        if y_scaling is not None:
            ax.set_ylim(y_scaling)  # Y-axis scaling
        # ax.set_ylim([y_min,y_max]) # Y-achis auto-scaling

        # add several helplines if wanted
        if baseline is True:
            ax.plot([210, x_max], [0, 0], color='k', linestyle='-', linewidth=1)  # plots baseline into graph
        if line1 is not None:
            ax.plot([210, x_max], line1, color='k', linestyle=':', linewidth=1)  # plots line into graph
        if line2 is not None:
            ax.plot([210, x_max], line2, color='k', linestyle=':', linewidth=1)
        if line3 is not None:
            ax.plot([210, x_max], line3, color='k', linestyle=':', linewidth=1)
        if line4 is not None:
            ax.plot([210, x_max], line4, color='k', linestyle=':', linewidth=1)
        if line5 is not None:
            ax.plot([210, x_max], line5, color='k', linestyle=':', linewidth=1)

        # make it nice
        ax.legend(rows)
        ax.set_xlabel(x_label)  # Add an x-label to the axes.
        ax.set_ylabel(y_label)  # Add a y-label to the axes.

        if len(subtitle) < (c - 1):
            ax.set_title("Fig. " + ascii_lowercase[c - 1])
        else:
            ax.set_title(subtitle[c - 1])
        c = c + 1

    # Title option
    fig.suptitle(title, fontsize=16)

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def function3d(*df, x_min=None, x_max=None):
    """Plots 3d function.
     In progress.
     """

    plt.ion()
    fig = plt.figure()  # create figure
    c = 1  # control variable

    for i in df:
        ax = fig.add_subplot(111, projection='3d')

        if x_min is None:
            x_min = list(i.index)[0]
        if x_max is None:
            x_max = list(i.index)[-1]
        x = list(df.loc[x_min:x_max].index)

        col2 = i.columns[-1]
        col1 = i.columns[0]
        y = i.loc[x_min:x_max, [col2]].to_numpy()  # get y-values
        z = i.loc[x_min:x_max, [col1]].to_numpy()  # get z-vaalues
        y, z = y.flatten(), z.flatten()
        ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        c = c + 1

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )
