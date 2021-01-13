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










