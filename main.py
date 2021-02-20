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
import analise as ana


def excel_worksheet(c_data):
    """ Transforms data to Excel.

    Creates for each array in 'CData.data' and 'CData.temp_value' an
    excel worksheet in an excel document.

    Parameter:
    ---------
        c_data: CData object from cdata
            measurement data

    Returns:
    -------
        excel_path: string
            path of the excel document
    """
    # getting name through input and creating path for ExcelWriter
    name = input('Insert name for Excel document including \".xlsx\"')
    excel_path = os.path.join(*c_data.path, name)
    # specify column names and units
    # header = [["Wavelength", "CD", "HT", "Absorption"],
    #           ["nm", "mdeg", "V", ""]
    #          ]
    header = ["Wavelength", "CD", "HT", "Absorption"]

    # creating excel document and let it open while adding sheets
    with pd.ExcelWriter(excel_path) as writer:
        for key in c_data.data:
            df = pd.DataFrame(c_data.data[key], columns=header)
            df = df.set_index("Wavelength")
            df.to_excel(writer, sheet_name=str(key), index=True)

        # add Wavelengths_Temperature Matrix as extra sheet
        df = c_data.temp_val()
        df.to_excel(writer, sheet_name="temp_matrix")

        # adding correlation matrices as sheets
        """arr1, arr2 = correlation(df)
        df1 = pd.DataFrame(arr1)
        df2 = pd.DataFrame(arr2)
        plot.heatmap_plot(df2)
        df1.to_excel(writer, sheet_name="Sync_Spec")
        df2.to_excel(writer, sheet_name="Async_Spec")"""

    return excel_path


def open_all(p):
    """open all folder of path 'p'"""
    folders = os.listdir(p)     # get folders as list
    data_all = {}
    for i in folders:           # get each folders 3 first words as name
        name = i.split(' ')
        name = ''.join(name[0:3])
        data_all.update({name: cdata.CData(folders)})   # then create object


# main function which is started when program run
if __name__ == '__main__':
    # User input for files
    print("Please insert the full directory of the Data: ")

    path = input()
    # test  stuff
    data = cdata.CData(path)
    print(data.t_list)
    plot.heatmap( data.t_df)
    # test
