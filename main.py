"""
Transfering CD Data to Excel format

This program opens each file from a given directory, transforms the relevant data to an array
and adds it as new worksheet to a newly created excel document.

Version 1.0

contact: Christoph Hadlich, christoph.hadlich@posteo.de
"""

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def file_opening(filename):
    "Opens the file filename and returns relevant data  in a matrix form"

    file = open(filename, "r")
    list_of_lines = file.readlines()

    # Parameter for head and tail part of document which will be deleted
    head = 21
    tail = 152
    del list_of_lines[tail:]
    del list_of_lines[:head]

    # separating numbers into sublist
    for i in range(len(list_of_lines)):
        list_of_lines[i] = list_of_lines[i].replace("\n","")
        list_of_lines[i] = list_of_lines[i].replace(",",".")
        list_of_lines[i] = list_of_lines[i].split('\t',3)

    # transform list of lists to array of float-type
    matrix = np.array(list_of_lines)
    matrix = matrix.astype(np.float)
    file.close()

    return matrix

def PertubationMatrix (files, directory):
    """ summorizes the CD values from a list of files into a matrix according to wavelength and
     pertubation written in filesname e.g. temperature
     Input_ LIST files, STRING directory
     Ouput: DATAFRAME PertubMatrix
     """

    # Define max. and min. values from measured wavelengths in nm
    wave_max = 330
    wave_min = 200

    # create List of measured wavenlength
    wave_list =[wave_min]
    for i in range(wave_max - wave_min):
        wave_list.append(wave_min+i+1)

    # create PertubMatrix
    PertubMatrix = pd.DataFrame({'Wavelength': wave_list})

    # loop for each file
    for i in range(len(files)):
        path = os.path.join(directory, files[i])

        # getting temperature from filesname
        temp = files[i]
        if temp[-7] == '.':
            temp = temp[-9:-4]
        else:
            temp = temp[-6:-4]

        # getting data as matrix and pass it on to excel
        df = pd.DataFrame(file_opening(path), columns=['Wavelength', 'HT', temp, 'Absorption'])

        # Delete three columns and change name of last
        df = df.drop(df.columns[[1, 3]], axis=1)
        PertubMatrix = pd.merge(PertubMatrix, df, on='Wavelength')

    # Set Wavelength as Index
    PertubMatrix.set_index(keys='Wavelength', inplace=True)

    return PertubMatrix

def correlation (Exp_Spectrum):
    """ calculates the Synchronous and Asynchronous 2D correlation of a given Spectrum.
    Hereby the dynamic spectrum is taken by substracting the average.
    Input:  Dataframe Matrix of measured Spectrum
    Output: Array Synchronous Spectrum, Arr Asynchronous Spectrum
     """
    # calculating average and dynamic spectrum as numpy array
    Average = Exp_Spectrum.mean(axis=1).to_numpy()
    Dyn_Spectrum = Exp_Spectrum.to_numpy() - Average[:,None]

    # getting number of rows and columns
    rows = Dyn_Spectrum.shape[0]
    cols = Dyn_Spectrum.shape[1]

    # creating 2d arrays for sync and async spectra
    size = (rows, rows)
    Sync_Spectrum = np.zeros(size)
    Async_Spectrum = np.zeros(size)

    # creating Hilbert_Noda_matrix for async spectrum
    arr = np.arange(1, rows+1)
    H_N_Matrix = arr - np.array([arr]).T + np.identity(rows)
    H_N_Matrix = 1/(np.pi*H_N_Matrix)
    H_N_Matrix = (H_N_Matrix-H_N_Matrix.T)/2
    H_N_Matrix = H_N_Matrix[...,:cols]

    # calculating sync and async values for each row and column
    # Work_Note: maybe reduce calculation due to symmetry?
    for i in range(rows):
        for k in range(rows):

            Sync_Spectrum[i,k] = np.sum(Dyn_Spectrum[i]*Dyn_Spectrum[k])/(cols-1)
            Async_Spectrum[i,k] = np.sum(Dyn_Spectrum[i]*np.sum(np.matmul(H_N_Matrix,Dyn_Spectrum[k])))/(cols-1)

    # returns Spectra; maybe change later Dtype back to DataFrame
    return Sync_Spectrum, Async_Spectrum


def Heatmap_plot(df):
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



def excel_worksheet (files, directory):
    " creates for each file in LIST files a excel worksheet in a new excel document"
    name = input('Insert name for Excel document including \".xlsx\"')

    # specify column names and units
    #header = [["Wavelength", "CD", "HT", "Absorption"], ["nm", "mdeg", "V", ""]]
    header = ["Wavelength", "CD", "HT", "Absorption"]

    # creating excel document and let it opend while adding sheets
    with pd.ExcelWriter(name) as writer:

        # loop for each file
        for i in range(len(files)):
            path = os.path.join(directory, files[i])

            # getting temperature as sheetname from filesname
            sheetname = files[i]
            if sheetname[-7] == '.':
                sheetname = sheetname[-9:-4]
            else:
                sheetname = sheetname[-6:-4]

            # getting data as matrix and pass it on to excel
            df = pd.DataFrame(file_opening(path), columns=header)
            df.to_excel(writer, sheet_name=sheetname, index=False)

        # add Wavelengths_Temperature Matrix as extra sheet
        df = PertubationMatrix(files, directory)
        df.to_excel(writer, sheet_name="temp_matrix")

        arr1, arr2 = correlation(df)
        df1 = pd.DataFrame(arr1)
        df2 = pd.DataFrame(arr2)
        Heatmap_plot(df1)
        df1.to_excel(writer, sheet_name="Sync_Spec")
        df2.to_excel(writer, sheet_name="Async_Spec")

    return name


# main function which is started when program run
if __name__ == '__main__':
    # User input for files
    print("Please insert the full directory of the Data: ")
    directory = input()
    list_of_files = os.listdir(directory)

    # opening function to create excel worksheets and return final path
    document = excel_worksheet(list_of_files, directory)
    final_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), document)
    print("Document successful saved under: \t" + final_path)










