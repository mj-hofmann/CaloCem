import os
import re
import pandas as pd

"""
read data from xls-file

Parameters
----------
file : str
    filepath.
show_info : bool, optional
    flag whether or not to show information. The default is True.

Returns
-------
data : pd.DataFrame
    data contained in file
"""

# data path
path = os.getcwd() + os.sep + os.pardir + os.sep + "DATA"

# init list of DataFrames
list_of_dfs = []

# loop
for _f in os.listdir(path):
    # ckeck xls. files
    if not _f.endswith(".xls"):
        # go to next
        continue

    # info
    print(_f)

    # new var
    file = path + os.sep + _f

    # define Excel file
    xl = pd.ExcelFile(file)

    # parse "data" sheet
    df_data = xl.parse("Raw data", header=None)

    # replace init timestamp
    df_data.iloc[0, 0] = "time"

    # get new column names
    new_columnames = []
    for i, j in zip(df_data.iloc[0, :], df_data.iloc[1, :]):
        # build
        new_columnames.append(
            re.sub("[\s\n\[\]\(\)° _]+", "_", f"{i}_{j}".lower())
            .replace("/", "_")
            .replace("_signal_", "_")
        )

    # set
    df_data.columns = new_columnames

    # cut out data part
    df_data = df_data.iloc[2:, :].reset_index(drop=True)

    # drop column
    try:
        df_data = df_data.drop(columns=["time_markers_nan"])
    except KeyError:
        pass

    # remove columns with too many NaNs
    df_data = df_data.dropna(axis=1, thresh=3)
    # # remove rows with NaNs
    # df_data = df_data.dropna(axis=0)

    # float conversion
    for _c in df_data.columns:
        # convert
        df_data[_c] = df_data[_c].astype(float)

    # add sample information
    df_data["sample"] = file

    # append to list
    list_of_dfs.append(df_data)

    # break

# to overall df
data = pd.concat(list_of_dfs)

# aux column
data["sample_short"] = [os.path.basename(i) for i in data["sample"]]

print(data.groupby(by="sample_short").last()["time_s"])
