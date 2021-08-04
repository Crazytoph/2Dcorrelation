"""Module for pre-treating CD-Data

This module contains the class definition for creating an object from
CD Data single files for a specific DNA origami type and Denaturant at fixed
concentration and different temperatures and collects useful methods for
displaying first aspects. It was designed for Data of the JASCO 8015
CD_Spectrometer and Devices with similar output.

Classes:
-------
CData
"""

import numpy as np
import pandas as pd
import os


class CData:
    """A class to represent whole Data of one CD measurement.

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
        t_list: list
            list of all measured temperatures
        cd_df: DataFrame
            df of all CD-values depending on wavelength and temperature
        absorb_df: DataFrame
            df of all Absorbance-values depending on wavelength and temperature

    Methods:
    -------
        cd_df():
            returns CD values in a Dataframe for wavelengths and temperature
        absorb_df():
            returns absorbance values in a DataFrame for wavelengths and temperature
        __path_split():
            split path into folder names
        __folder_opening():
            opens each file in folder
        --file_opening():
            extracts data from file
    """

    def __init__(self, path):
        """Constructs all necessary attributes of the Data

        Parameters:
        _________

        path: string
            path of data

        Notes:
        -----
        Access type (public, protected, private) needs still to be defined.
        """
        self.path = path
        self.dna = self.__name_split()[-4]
        self.origami = self.__name_split()[-3]
        self.denaturant = self.__name_split()[-1]
        self.concentration = self.__name_split()[-2]
        self.data = self.__folder_opening()
        self.t_list = list(self.data.keys())
        self.absorb_df = self.absorb_df()
        self.cd_df = self.cd_df()
        self.ht_df = self.ht_df()

    def absorb_df(self):
        """Creates wavelength-temperature dataframe.

        Takes the data dictionary and takes the Absorbance value in relation to
        the wavelength as index and the respective temperature as column
        value

        Returns:
        ---------
        absorb_matrix: data-frame
            index is the wavelengths, columns the temperature
        """
        # Define max. and min. values from measured wavelengths in nm
        wave_max = 330
        wave_min = 200

        # create List of measured wavelength
        wave_list = [wave_min]
        for i in range(wave_max - wave_min):
            wave_list.append(wave_min + i + 1)

        # create DataFrame
        absorb_matrix = pd.DataFrame(wave_list, columns=[0])

        # iterate through dictionary adding columns to dataframe
        for key in self.data:
            col = self.data[key]
            col = col[..., [0, 3]]
            df = pd.DataFrame(col, columns=[0, key])
            absorb_matrix = pd.merge(absorb_matrix, df, on=0)

        # order columns, change name to wavelength as index
        cols = absorb_matrix.columns
        absorb_matrix = absorb_matrix[cols.sort_values()]
        absorb_matrix = absorb_matrix.rename(columns={0: 'wavelength'})
        absorb_matrix = absorb_matrix.set_index('wavelength')

        return absorb_matrix

    def cd_df(self):
        """Creates wavelength-temperature dataframe.

        Takes the data dictionary and takes the CD value in relation to
        the wavelength as index and the respective temperature as column
        value

        Returns:
        ---------
        cd_matrix: data-frame
            index is the wavelengths, columns the temperature
        """
        # Define max. and min. values from measured wavelengths in nm
        wave_max = 330
        wave_min = 200

        # create List of measured wavelength
        wave_list = [wave_min]
        for i in range(wave_max - wave_min):
            wave_list.append(wave_min + i + 1)

        # create DataFrame
        cd_matrix = pd.DataFrame(wave_list, columns=[0])

        # iterate through dictionary adding columns to dataframe
        for key in self.data:
            col = self.data[key]
            col = col[..., 0:2]
            df = pd.DataFrame(col, columns=[0, key])
            cd_matrix = pd.merge(cd_matrix, df, on=0)

        # order columns, change name to wavelength as index
        cols = cd_matrix.columns
        cd_matrix = cd_matrix[cols.sort_values()]
        cd_matrix = cd_matrix.rename(columns={0: 'wavelength'})
        cd_matrix = cd_matrix.set_index('wavelength')

        # change unit into molar ellipticity, according to
        # https://www.photophysics.com/circular-dichroism/chirascan-technology
        # /circular-dichroism-spectroscopy-units-conversions/
        absorbance = self.absorb_df.loc[260, 20]
        c = absorbance / (0.02 * 0.1)  # [µg/mL]
        molar_c = c * 10 ** (-6) / (4472760.4 * 10 ** (-3))  # [mol/l]
        cd_matrix = 100 * cd_matrix / (molar_c * 0.1)     # pathway 1mm in cm

        return cd_matrix

    def ht_df(self):
        """Creates wavelength-temperature dataframe.

        Takes the data dictionary and takes the HT value in relation to
        the wavelength as index and the respective temperature as column
        value

        Returns:
        ---------
        ht_matrix: data-frame
            index is the wavelengths, columns the temperature
        """
        # Define max. and min. values from measured wavelengths in nm
        wave_max = 330
        wave_min = 200

        # create List of measured wavelength
        wave_list = [wave_min]
        for i in range(wave_max - wave_min):
            wave_list.append(wave_min + i + 1)

        # create DataFrame
        ht_matrix = pd.DataFrame(wave_list, columns=[0])

        # iterate through dictionary adding columns to dataframe
        for key in self.data:
            col = self.data[key]
            col = col[..., [0, 2]]
            df = pd.DataFrame(col, columns=[0, key])
            ht_matrix = pd.merge(ht_matrix, df, on=0)

        # order columns, change name to wavelength as index
        cols = ht_matrix.columns
        ht_matrix = ht_matrix[cols.sort_values()]
        ht_matrix = ht_matrix.rename(columns={0: 'wavelength'})
        ht_matrix = ht_matrix.set_index('wavelength')

        return ht_matrix

    def __name_split(self):
        """Splits name into list and formats it to get key information."""

        # split and change filename into type list
        name = os.listdir(self.path)[-1]
        name = os.path.split(name)[1]
        name_split = name.split(' ')
        name_split[-1] = name_split[-1].split('-')[0]
        name_split[-3: -1] = [''.join(name_split[-3: -1])]
        # name_split.remove('mit')
        return name_split

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
            # getting repository
            repository = os.path.join(self.path, files[i])

            # checking whether it is really a file
            if os.path.isfile(repository) is False:
                continue

            # getting temperature from filenames#
            temp = files[i]
            if temp[-7] == '.':
                temp = temp[-9:-4]
            else:
                temp = temp[-6:-4]

            # round temp to int
            exact = float(temp)
            up = np.ceil(exact)
            down = np.floor(exact)
            if np.abs(up - exact) <= np.abs(exact - down):
                temp = int(up)
            else:
                temp = int(down)

            # add file to dictionary
            data[temp] = self.__file_opening(repository)

        return data

    @staticmethod
    def __file_opening(filename):
        """Extract relevant data from file.

        This function opens a given file in .txt-format and returns the
        measured data as 'array'. Specialized on the output of the JASCO 8015
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
