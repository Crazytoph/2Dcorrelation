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


# main function which is started when program run
if __name__ == '__main__':
    # User input for files
    print("Please insert the full directory of the Data: ")

    path = input()
    # test  stuff
    data = cdata.CData(path)

    plot.heatmap(data.t_df, data.t_df)
