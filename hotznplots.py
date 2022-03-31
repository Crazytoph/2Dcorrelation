"""Module for the hot plots.

Functions:
---------
    heatmap(*data, x_min=[], x_max=[], y_min=[], y_max=[], swap=True,
            c_min=[], c_max=[], x_label="temperature[°C]",
            y_label="wavelength[nm]", title="Heatmap", subtitle=[], contour_lines="True"
            ):
        Plots heatmap of an dataframe.
    mult_func(rows, *probes, error={}, swap=False,
              x_label="Temperature[°C]", y_label="CD values[mdeg]", title="", subtitle=[],
              backgroundcolor="white", marker=[], linestyle=[], label=None,
              y_scaling=False, y_min=[], y_max=[], baseline=False,  x_min=None, x_max=None, vertical_line=[]
              ):
        Plots one or multiple subplots.
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
    function3d(*df, x_min=None, x_max=None):
        Plots 3d function. Work in progress.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from string import ascii_lowercase


def heatmap(*data, x_min=[], x_max=[], y_min=[], y_max=[], swap=True,
            c_min=[], c_max=[], c_label="test", hline=None, vline=None, x_label="temperature / °C",
            y_label="wavelength / nm", title="Heatmap", subtitle=[], contour_lines=None, backgroundcolor='white'
            ):
    """Plots heatmap.
       This function plots heatmaps of DataFrames 'df' from in given area ('x_min,
       'x_max, 'y_min', y_max')
       Parameters:
       ----------
        *data: DataFrames or CData-object  -> * indicates the possibility of multiple df
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
        title, x_label, y_label, c_label: string
           description for plot
        hline, vline: int
            values for a possible horizontal/vertical line
        subtitle: list of strings
            titles for all subfigures
        contour_lines: list
            list for possible contour_lines

       Notes:
       -----
       To-Do:  defining map-style
       """
    # ensure  we are interactive mode
    # this is default but if this notebook is executed out of order it may have been turned off
    plt.ion()
    fig = plt.figure(facecolor=backgroundcolor)

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
        if len(data) < 2:
            width = len(data)
        else:
            width = 2
        ax = fig.add_subplot((len(data) // 3) + 1, width, c)
        arr = pd.DataFrame.to_numpy(df.loc[x_min[c-1]:x_max[c-1], y_min[c-1]:y_max[c-1]])

        # if axis should be swapped
        if swap:
            arr = arr.T
            extent = extent[2:4] + extent[0:2]
            xlabel, ylabel = y_label, x_label
        else:
            xlabel, ylabel = x_label, y_label

        # get colorscale range if not given
        if len(c_min) < c:
            c_min.append(arr.min())
        if len(c_max) < c:
            c_max.append(arr.max())

        # here happens the 'real' plot with all settings
        im = ax.imshow(arr, aspect='auto', cmap='jet',
                       vmax=c_max[c-1], vmin=c_min[c-1], interpolation='bicubic',
                       extent=extent, origin="lower")

        # add contour lines of wanted
        if contour_lines is not None:
            CS = ax.contour(arr, contour_lines, alpha=0.8, colors='k', extent=extent,
                            linestyles=(["dashed"] * (len(contour_lines) // 2) + ["dotted"] +
                                        ["dashed"] * (len(contour_lines) // 2)))
            # Negative contours default to dashed.
            #ax.clabel(CS, fontsize=9, inline=True) # label the contour lines

        # add label
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        if hline is not None:
            # for melting temp: linestyle="dotted", lindewidth=4
            ax.axhline(y=hline, color='black', linewidth=2, linestyle='dotted', label='Melting Temperature')
        if vline is not None:
            ax.axvline(x=hline, color='black', linewidth=2, linestyle='solid', label='Melting Temperature')

        # set fontsize ticks
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # set visibility of x-axis as False
        #xax = ax.axes.get_xaxis()
        #xax = xax.set_visible(False)

        # set visibility of y-axis as False
        #yax = ax.axes.get_yaxis()
        #yax = yax.set_visible(False)
        # change ticklabes *special*
        """ax.set_xticks([0, 15, 33, 50, 70, 87])
        ax.set_xticklabels([215, 230, 247, 265, 285, 'Absorbance'])
        ax.set_yticks([0, 15, 33, 50, 70, 87])
        ax.set_yticklabels([215, 230, 247, 265, 285, 'Absorbance'])"""

        # add subtitle
        if len(subtitle) < c:
            ax.set_title("Fig. " + ascii_lowercase[c - 1], fontsize=16)
        else:
            ax.set_title(subtitle[c - 1], fontsize=16)
        
        # add colorbar for each subfigur
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax2)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(c_label, fontsize=16, rotation=90)
        c = c + 1  # push control variable

    # Title option
    fig.suptitle(title, fontsize=21)
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
             x_label="Temperature[°C]", y_label="CD values[molar mdeg]", title="", subtitle=[],
             y_scaling=False, baseline=False
             ):
    """Plots simple graph of DataFrames
    This function plots several CD values for each selected Temperature Column
    'r'
    Parameters:
    ----------
        *df: Tuple of DataFrames -> * indicates the possibility of multiple df
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
        ax = fig.add_subplot(1, len(df), c)  # create subplot

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
            ax.plot(x, y, linestyle='-', marker='', color=colors[k % 10],
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

    # Title option
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(w_pad=0.5)

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def mult_func(rows, *probes, error={}, swap=False,
              x_label="temperature / °C", y_label=r"molar CD "r"$/ \ \frac{deg \times cm^{2}}{dmol}$",
              title="", subtitle=[], backgroundcolor="white", marker=[], linestyle=[], label=None,
              y_scaling=False, y_min=[], y_max=[], baseline=False,  x_min=None, x_max=None, vertical_line=[],
              sec_ax=None, colors=None
              ):
    """Plots one or multiple graphs of DataFrames or CData-Objects.

    This function plots for each value in 'rows' a graph from each object in 'probes'.
    Thereby it creates for each list a new subplot and puts all elements of one list into it.
    List parameter like 'y_min' should contain one value for each subplot.

    Parameters:
    ----------
        rows: list of index-values
            rows of DataFrames which should be plotted
        *probes: Tuple or List of cdata.CData objects or DataFrames -> * indicates the possibility of multiple df
            list containing data for one subplot
        error: dictionary of graph-error couple
            error vales for graphs, Default: empty
        swap: boolean
            parameter determining whether to change axes, Default: True
        x_label, y_label,title: string
            description for plot, Default: "temperature /°C", "molar cd / mdeg /cm^2 dmol", ""
        subtitle: list of strings
            title for each subplot, Default" empty
        backgroundcolor: string-code
             backgroundcolor for plot, Default: "white"
        marker: list of string-code
            marker-style for each subplot, Default: empty
        linestyle: list of string-code
            line-style for each subplot, Default: empty
        label: None or list of string
            possible label names, only one possibility for all subplots, Default: empty
        y_scaling: boolean
            parameter determining whether to scale y-axis, Default: False
        y_min, y_max: list of int
            min and max value of data on y-axis for each subplot, Default: empty
        baseline: boolean
            parameter determining whethter to add baseline, Default: False
        x_min, x_max: int
             min and max value for baseline, Default: empty
        vertical_line: list of float or in
            values where to add vertical lines, labeld as "melting temperature", Default: empty
        sec_ax: list
            values for second y axis

    Returns:
    -------
        None  but plots.

    Notes:
    -----
    To-Do:  defining line-style,
    """
    # create color list and color variables
    if colors is None:
        colors = ['tab:blue', 'tab:orange',  'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan']
        #colors = ['black', 'firebrick', 'lightcoral', 'red',  'darkred']    # red for publication
        #colors = ['midnightblue', 'cornflowerblue', 'darkorange', 'aquamarine',  'blue', ]

    c = 1  # control variable

    plt.ion()  # needed for jupyter
    fig = plt.figure(facecolor=backgroundcolor)  # create figure

    # iterate through all lists for each subfigure
    for subs in probes:
        # create subplot and array
        if len(probes) < 4:
            width = len(probes)
        else:
            width = 3

        ax = fig.add_subplot((len(probes) // 3)+1, width, c)  # create subplot
        ax.set_facecolor(backgroundcolor)
        # iterate through all data to b plotted into one subfigure

        k = 0  # control variable
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

                x = np.array(df.loc[:, x_min:x_max].columns)  # get x-values
                y = pd.DataFrame.to_numpy(df.loc[r, x_min:x_max])  # get y-values
                x = x[~(pd.isna(y))]          # remove NaN values
                y = y[~(pd.isna(y))]

                ax.plot(x, y,
                        linestyle=linestyle[k], marker=marker[k],
                        color=colors[k % 10], linewidth=2, label=r
                        )

                # plot errorbar if given
                if r in error.keys():
                    yerr = error[r]
                    ax.errorbar(x, y, yerr=yerr, color=colors[k % len(colors)], alpha=0.6)
                    # also create fill
                    #ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[k % len(colors)])
                k = k + 1

        if len(vertical_line) >= c:
            ax.axvline(x=vertical_line[c-1], color='tab:red', linestyle='--', linewidth=2,
                       label='melting Temp')

        # scaling
        if y_scaling is True:
            ax.set_ylim([y_min[c-1], y_max[c-1]])  # Y-axis scaling

        # baseline
        if baseline is True:
            ax.plot([x_min, x_max], [0, 0], color='k', linestyle=':', linewidth=1)

        # set fontsize ticks
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)


        # make it nice
        ax.set_xlabel(x_label, fontsize=16)  # Add an x-label to the axes.
        ax.set_ylabel(y_label, fontsize=16)  # Add a y-label to the axes.


        if sec_ax is not None:
            # put second y_axis for "GdmCl_0.5M" normed
            # function to bring back values
            def expand_back(y):
                return (y - sec_ax[0]) / (sec_ax[1] - sec_ax[0])

            # inverse function
            def norm(y):
                return y * (sec_ax[1] - sec_ax[0]) + sec_ax[0]

            # put axis and label
            secax = ax.secondary_yaxis('right', functions=(expand_back, norm))
            secax.tick_params(labelsize=16)
            secax.set_ylabel('ssDNA fraction / %', fontsize=16)

        if label is None:  # add legend
            ax.legend(fontsize=16)
        else:
            ax.legend(label, fontsize=16)

        # add subtitle
        if len(subtitle) < c:
            ax.set_title("Fig. " + ascii_lowercase[c - 1], fontsize=16)
        else:
            ax.set_title(subtitle[c - 1], fontsize=16)
        c = c + 1

    # Title option
    fig.suptitle(title, fontsize=21)
    plt.tight_layout()


    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )


def functionT(rows, *df, x_min=None, x_max=None, y_min=None, y_max=None, swap=True,
              x_label="Wavelength[nm]", y_label="CD values[molar mdeg]", title="", subtitle=[],
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


def function3d(data, title="3D-Plot", label=None, x_label="Wavelength [nm]", y_label="Temperature [°C]",
               z_label=r"CD- Values "r"$[\frac{deg \times cm^{2}}{dmol}]$", backgroundcolor='white'):
    """Plots 3d function.
     In progress.
     """
    if not isinstance(data, pd.core.frame.DataFrame):
        df = data.cd_df
    else:
        df = data

    plt.ion()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, facecolor=backgroundcolor)
    ax.set_facecolor(backgroundcolor)
    # Make data.
    y, x = np.meshgrid(df.columns.astype(float), df.index)
    z = df.values

    """# Normalize to [0,1]
    norm = plt.Normalize(z.min(), z.max())
    colors = cm.jet(norm(z))
    rcount, ccount, _ = colors.shape

    # plot wireframe
    surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0, 0, 0, 0))"""

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, shade=True, cmap="jet", linewidth=0.1)
    surf.set_edgecolors(surf.to_rgba(surf._A))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.7, aspect=5)

    # optional: delete background
    """fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False"""

    # make it nice
    ax.set_xlabel(x_label, fontsize=16, labelpad=10)  # Add an x-label to the axes.
    ax.set_ylabel(y_label, fontsize=16,  labelpad=10)  # Add a y-label to the axes.
    ax.set_zlabel(z_label, fontsize=16,  labelpad=10, rotation=180)  # Add a z-label to the axes.

    # set fontsize ticks
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    if label is None:  # add legend
        ax.legend(fontsize=16)
    else:
        ax.legend(label, fontsize=16)

    # Title option
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # stuff for jupyter copied from 'https://github.com/matplotlib/ipympl'
    widgets.AppLayout(
        center=fig.canvas,
        footer=widgets.Button(icon='check'),
        pane_heights=[0, 6, 1]
    )
