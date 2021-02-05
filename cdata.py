"""Module for pre-treating CD-Data

This module contains the class definition for creating an object from
CD Data single files for a specific DNA origami type and Denaturant at fixed
concentration and different temperatures and collects useful methods for
displaying first aspects. It was designed for Data of the JASCO 8015
CD_Spectrometer and Devices with similar output.

Classes:
-------
CircData
"""

import numpy as np
import pandas as pd
import os


class CircData:
    """A class to represent Data of one CD measurement.

    An object of this class contains the whole data of one measurement of the
    JASCO 8015 CD_Spectrometer. For each temperature step, values are saved
    in array and combined with temperature in the dictionary 'data'. Other
    parameters represent the sample type defined in through the repository.
    The methods do small data treatment and simple calculations.

    Attributes:
    ----------
    path: string
        path of the original measurement files
    dna: string
        type of DNA measured
    denaturant:  string
        type of denaturant used
    concentration: string
        denaturant concentration used
    data: dictionary
        measured values for each temperature step
    temp: list
        list of all measured temperatures

    Methods:
    -------
    temp_val():
        returns CD values in a Dataframe for wavelengths and temperature
    """

    def __init__(self, reps):
        """Constructs all necessary attributes of the Data

        Parameters:
        _________

        reps: list
            folder names as string in order of sub-folder hierarchy

        Example:
        -------
        Path-List should be like: ["Rest", "DNA-type","denaturant",
        "concentration"]

        Notes:
        -----
        Access type (public, protected, private) needs still to be defined.
        """
        self.path = reps
        self.dna = self.__path_split()[-3]
        self.denaturant = self.__path_split()[-2]
        self.concentration = self.__path_split()[-1]
        self.data = self.__folder_opening()
        self.temp = list(self.data.keys())

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

    def __path_split(self):
        """Splits path into a list of folder names."""
        # split path into single folder names as list
        path_split = []
        path = self.path
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                path_split.insert(0, parts[0])
                break
            elif parts[1] == path:  # sentinel for relative paths
                path_split.insert(0, parts[1])
                break
            else:
                path = parts[0]
                path_split.insert(0, parts[1])

        return path_split

    def __folder_opening(self):
        """Opens folder for given repository and summarizes data.

        Look into folder under 'reps' and opens each file with 'file_opening'
        combing the returned data-array with the rounded measured temperature.

        Returns:
        -------
            data: dictionary
                each array as value with the measured temperature as key
        """
        # getting list of files and create empty dictionary
        files = os.listdir(self.path)
        data = {}

        # loop for each file
        for i in range(len(files)):
            # getting temperature from filenames
            temp = files[i]
            if temp[-7] == '.':
                temp = temp[-9:-4]
            else:
                temp = temp[-6:-4]
            # round temp to int
            exact = float(temp)
            up = np.ceil(exact)
            down = np.floor(exact)
            if np.abs(up - exact) < np.abs(exact - down):
                temp = int(up)
            else:
                temp = int(down)

            # adding values to data
            repository = os.path.join(self.path, files[i])
            data[temp] = self.__file_opening(repository)

        return data

    def __file_opening(self, filename):
        """Extract relevant data from file.

        This function opens a given file in .txt-format and returns the measured
        data as 'matrix'. Specialized on the output of the JASCO 8015
        CD_Spectrometer.

        Parameter:
        ---------
            filename: string
                name of the file

        Returns:
        -------
            matrix: array-like
                Data
        """
        file = open(filename, "r")
        list_of_lines = file.readlines()

        # Parameter for head and tail part of document which will be deleted
        head = 21
        tail = 152
        del list_of_lines[tail:]
        del list_of_lines[:head]

        # separating numbers into sublist
        for i in range(len(list_of_lines)):
            list_of_lines[i] = list_of_lines[i].replace("\n", "")
            list_of_lines[i] = list_of_lines[i].replace(",", ".")
            list_of_lines[i] = list_of_lines[i].split('\t', 3)

        # transform list of lists to array of float-type
        matrix = np.array(list_of_lines)
        matrix = matrix.astype(np.float)
        file.close()

        return matrix