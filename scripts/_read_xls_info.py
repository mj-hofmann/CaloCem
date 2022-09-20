import os
import pandas as pd

"""
read information from xls-file

Parameters
----------
file : str
    filepath.
show_info : bool, optional
    flag whether or not to show information. The default is True.

Returns
-------
info : pd.DataFrame
    information (metadata) contained in file

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

    # specify Excel
    xl = pd.ExcelFile(file)

    try:
        # read 'Experiment info' sheet
        info = (
            xl.parse("Experiment info", header=None, names=["parameter", "value"])
            .dropna(subset=["parameter"])
            .set_index("parameter")
            .T.set_index("Name")
        )

    except ValueError as e:
        # info
        print(f"{e} in file {_f}")
        # go to next
        continue

    list_of_dfs.append(info)


# get overall DataFrame
info = pd.concat(list_of_dfs, axis=0).reset_index()

# rename columns
info.columns = [c.lower().replace(" ", "_") for c in info.columns]

# clean Name info
info["name"] = [i if type(i) is str else " | ".join(i) for i in info["name"]]
