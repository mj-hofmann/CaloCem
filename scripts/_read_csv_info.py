import os
import pandas as pd

"""
read info from csv file

Parameters
----------
file : str
    filepath.

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
    if not _f.endswith(".csv"):
        # go to next
        continue

    # info
    print(_f)

    # new var
    file = path + os.sep + _f

    # define Excel file
    info = pd.read_csv(
        file,
        header=None,
        sep="No meaningful separator",
        engine="python",
        names=["full"],
    )

    # get "column" count
    info["count"] = [len(i) for i in info.iloc[:, 0].str.split(",")]

    # restrict to "reasonable" top-part
    info = info[(info.index < 50) & (info["count"] <= 2)]

    # identify row as "valid parameter row"
    info["valid_row"] = info["full"].str.match("^[^,][\w\s]+,")

    # restrict to valid rows
    info = info[info["valid_row"] == True]

    # extract parameter and corresponding value
    info["parameter"] = [i[0] for i in info["full"].str.split(",")]
    info["value"] = [i[1].strip('"') for i in info["full"].str.split(",")]

    # get revelant columns
    info = info[["parameter", "value"]].set_index("parameter").T

    # rename columns
    info.columns = [c.lower().replace(" ", "_") for c in info.columns]

    # set name as inde
    info = info.set_index("name")


# "get index back"
info = info.reset_index()

# clean Name info
info["name"] = [i if type(i) is str else " | ".join(i) for i in info["name"]]
