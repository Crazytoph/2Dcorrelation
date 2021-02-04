"""
Transfering CD Data to Excel format

This program opens each file from a given directory, transforms the relevant
data to an array and adds it as new worksheet to a newly created excel
document.

Version 1.0
"""

# importing modules
import numpy as np
import pandas as pd
import os
import cdata
import hotznplots as plot


def file_opening(filename):
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


def correlation(exp_spec):
    """ Performs 2D correlation analysis.

    Calculates the Synchronous and Asynchronous 2D correlation of a given
    Spectrum.
    Hereby the dynamic spectrum is taken by subtracting the average.

    Parameters:
    ----------
    exp_spec : Dataframe
            Matrix of measured Spectrum

    Returns:
    -------
    sync_spec, async_spec: array_like
            The synchronous respectively asynchronous
            correlation spectrum
    """
    # calculating average and dynamic spectrum as numpy array
    avg = exp_spec.mean(axis=1).to_numpy()
    dyn_spec = exp_spec.to_numpy() - avg[:, None]

    # getting number of rows and columns
    rows = dyn_spec.shape[0]
    cols = dyn_spec.shape[1]

    # creating 2d arrays for sync and async spectra
    size = (rows, rows)
    sync_spec = np.zeros(size)
    async_spec = np.zeros(size)

    # creating Hilbert_Noda_matrix for async spectrum
    arr = np.arange(1, rows + 1)
    h_n_m = arr - np.array([arr]).T + np.identity(rows)
    h_n_m = 1 / (np.pi * h_n_m)
    h_n_m = (h_n_m - h_n_m.T) / 2
    h_n_m = h_n_m[..., :cols]

    # calculating sync and async values for each row and column
    # Work_Note: maybe reduce calculation due to symmetry?
    for i in range(rows):
        for k in range(rows):
            sync_spec[i, k] = np.sum(dyn_spec[i]
                                     * dyn_spec[k]) \
                              / (cols - 1)
            async_spec[i, k] = np.sum(dyn_spec[i]
                                      * np.sum(np.matmul(h_n_m, dyn_spec[k]))) \
                               / (cols - 1)

    # returns Spectra; maybe change later Dtype back to DataFrame
    return sync_spec, async_spec


def excel_worksheet(circ_data):
    """ Transforms data to Excel.

    Creates for each array in 'circ_data.data' and 'circ_data.temp_value' an
    excel worksheet in an excel document.

    Parameter:
    ---------
        circ_data: CircData object from cdata
            measurement data

    Returns:
    -------
        excel_path: string
            path of the excel document
    """
    # getting name through input and creating path for ExcelWriter
    name = input('Insert name for Excel document including \".xlsx\"')
    excel_path = os.path.join(*circ_data.path, name)
    # specify column names and units
    # header = [["Wavelength", "CD", "HT", "Absorption"],
    #           ["nm", "mdeg", "V", ""]
    #          ]
    header = ["Wavelength", "CD", "HT", "Absorption"]

    # creating excel document and let it open while adding sheets
    with pd.ExcelWriter(excel_path) as writer:
        for key in circ_data.data:
            df = pd.DataFrame(circ_data.data[key], columns=header)
            df = df.set_index("Wavelength")
            df.to_excel(writer, sheet_name=str(key), index=True)

        # add Wavelengths_Temperature Matrix as extra sheet
        df = circ_data.temp_val()
        df.to_excel(writer, sheet_name="temp_matrix")

        # adding correlation matrices as sheets
        """arr1, arr2 = correlation(df)
        df1 = pd.DataFrame(arr1)
        df2 = pd.DataFrame(arr2)
        plot.heatmap_plot(df2)
        df1.to_excel(writer, sheet_name="Sync_Spec")
        df2.to_excel(writer, sheet_name="Async_Spec")"""

    return excel_path


def folder_opening(files_path):
    """Opens folder for given repository and summarizes data.

    Look into folder under 'reps' and opens each file with 'file_opening'
    combing the returned data-array with the rounded measured temperature.

    Parameters:
    ----------
        files_path: string
            path of the files
    Returns:
    -------
        data: dictionary
            each array as value with the measured temperature as key
    """
    # getting list of files and create empty dictionary
    files = os.listdir(files_path)
    data = {}

    # loop for each file
    for i in range(len(files)):
        repository = os.path.join(files_path, files[i])
        # getting temperature from filesname
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
        data[temp] = file_opening(repository)

    return data


# main function which is started when program run
if __name__ == '__main__':
    # User input for files
    print("Please insert the full directory of the Data: ")

    path = input()
    # split path into single folder names as list
    path_split = []
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

    #  recreating final path for documentary reasons reason
    final_path = os.path.join(*path_split)

    # test part
    data_in_dic = folder_opening(final_path) # getting data
    measurement1 = cdata.CircData(path_split, data_in_dic)  # creating object
    temperatures = measurement1.temp[::5] # getting temp values for plotting
    plot.function_plot(measurement1.temp_val(), temperatures, min=50,
                       limit= 100) #
    # plotting
