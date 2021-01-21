"""
Transfering CD Data to Excel format

This program opens each file from a given directory, transforms the relevant data to an array
and adds it as new worksheet to a newly created excel document.

Version 1.0

contact: Christoph Hadlich, christoph.hadlich@posteo.de
"""

# importing modules
import numpy as np
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
    """ calculates the average CD value per wavelength over temperature.
    Input:  Dataframe Matrix of Wavelength-Temperature
    Output: Dataframe one colume CD average per for each wavelength
     """
    Average = Exp_Spectrum.mean()
    Dyn_Spectrum = Exp_Spectrum - Average
    rows = Dyn_Spectrum.shape[0]
    cols = Dyn_Spectrum.shape[1]

    Sync_Spectrum = pd.DataFrame(index=Exp_Spectrum.index, columns=Exp_Spectrum.index)
    Async_Spectrum = pd.DataFrame(index=Exp_Spectrum.index, columns=Exp_Spectrum.index)

    arr = np.arange(1, rows+1)
    H_N_Matrix = arr - np.array([arr]).T + np.identity(rows)
    H_N_Matrix = 1/(np.pi*H_N_Matrix)
    H_N_Matrix = (H_N_Matrix-H_N_Matrix.T)/2


    for i in range(rows):
        for k in range(rows):

            Sync_Spectrum.iloc[i,k] = np.sum(Dyn_Spectrum.iloc[i]*Dyn_Spectrum.iloc[k], axis=0)/(cols-1)
            arr2 = Dyn_Spectrum.iloc[i]*Dyn_Spectrum.iloc[k].T/(cols-1)
    return arr2

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











