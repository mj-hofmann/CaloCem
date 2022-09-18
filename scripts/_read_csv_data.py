import os
import re
import pandas as pd

"""
read data from .csv file

Parameters
----------
file : str
    filepath.

Returns
-------
data : pd.DataFrame
    experimental data contained in file.

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
    df_data = pd.read_csv(
        file, header=None, sep="No meaningful separator", engine="python"
    )

    # get "column" count
    df_data["count"] = [len(i) for i in df_data[0].str.split(",")]

    # get most frequent count --> assume this for selection of "data" rows
    df_data = df_data.loc[
        df_data["count"] == df_data["count"].value_counts().index[0], [0]
    ]

    # init and loop list of lists
    list_of_lists = []
    for _, r in df_data.iterrows():
        # append to list
        list_of_lists.append(str(r.to_list()).strip("['']").split(","))

    # get DataFrame from list of lists
    df_data = pd.DataFrame(list_of_lists)

    # get new column names
    new_columnames = []
    for i in df_data.iloc[0, :]:
        # build
        new_columname = (
            re.sub('[\s\n\[\]\(\)° _"]+', "_", i.lower())
            .replace("/", "_")
            .replace("_signal_", "_")
            .strip("_")
        )

        # select appropriate unit
        if new_columname == "time":
            new_columname += "_s"
        elif "temperature" in new_columname:
            new_columname += "_c"
        elif new_columname == "heat_flow":
            new_columname += "_w"
        elif new_columname == "heat":
            new_columname += "_j"
        elif new_columname == "normalized_heat_flow":
            new_columname += "_w_g"
        elif new_columname == "normalized_heat":
            new_columname += "_j_g"
        else:
            new_columname += "_nan"

        # add to list
        new_columnames.append(new_columname)

    # set
    df_data.columns = new_columnames

    # cut out data part
    df_data = df_data.iloc[1:, :].reset_index(drop=True)

    # drop column
    try:
        df_data = df_data.drop(columns=["time_markers_nan"])
    except KeyError:
        pass

    # remove columns with too many NaNs
    df_data = df_data.dropna(axis=1, thresh=3)
    # # remove rows with NaNs
    df_data = df_data.dropna(axis=0)

    # float conversion
    for _c in df_data.columns:
        # convert
        df_data[_c] = df_data[_c].astype(float)

    # restrict to "time_s" > 0
    df_data = df_data.query("time_s >= 0")

    # add sample information
    df_data["sample"] = file

    # append to list
    list_of_dfs.append(df_data)


# to overall df
data = pd.concat(list_of_dfs)
