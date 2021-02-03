"""Module for pre-treating CD-Data

This module contains the class definition for creating an object from
CD Data single files for a specific DNA origami type and Denaturant at fixed
concentration and different temperatures and collects useful methods for
displaying first aspects. It was designed for Data of the JASCO 8015
CD_Spectrometer and Devices with similar output.

Classes:
-------

F
"""

import numpy as np
import pandas as pd


class CircData:
    """A class to represent Data of one measurement

    Summary

    Attributes:
    ----------

    Methods:
    -------
    """

    def __init__(self, path, data):
        """Constructs all necessary attributes of the Data

        Parameters:
        _________

            path: list
                folder names as string in order of sub-folder hierarchy
             files: dictionary
                all raw data from each measurement point as arrays linked to
                their respective temperature

        Example:
        -------
        Path-List should be like: ["repository", "DNA-type","denaturant",
        "concentration"]

        Notes:
        -----
        Access type (public, protected, private) needs still to be defined.
        """
        self.path = path
        self.dna = path[-3]
        self.denaturant = path[-2]
        self.concentration = path[-1]
        self.data = data

    def temp_val(self):
        """Creates wavelength-temperature dataframe.

        Takes the data dictionary and takes the CD value in relation to
        the wavelength as index and the respective temperature as column
        value

        Returns:
        ---------
            temp_matrix: data-frame
                index is the wavelengths, columns the temperature
        """
        # Define max. and min. values from measured wavelengths in nm
        wave_max = 330
        wave_min = 200

        # create List of measured wavelength
        wave_list = [wave_min]
        for i in range(wave_max - wave_min):
            wave_list.append(wave_min + i + 1)

        # create PertubMatrix
        temp_matrix = pd.DataFrame(wave_list, columns=[0])

        # iterate through dictionary adding columns to dataframe
        for key in self.data:
            col = self.data[key]
            col = col[..., 0:3:2]
            df = pd.DataFrame(col, columns=[0, key])
            temp_matrix = pd.merge(temp_matrix, df, on=0)

        # order columns, change name to wavelength as index
        cols = temp_matrix.columns
        temp_matrix = temp_matrix[cols.sort_values()]
        temp_matrix = temp_matrix.rename(columns={0: 'wavelength'})
        temp_matrix = temp_matrix.set_index('wavelength')

        return temp_matrix