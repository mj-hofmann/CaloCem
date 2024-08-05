# %%
import os
import pathlib
import sys

from CaloCem import tacalorimetry

# get path of script and set it as current path
pathname = os.path.dirname(sys.argv[0])
os.chdir(pathname)
filename = os.path.basename(__file__).replace(".py", "")

os.sys.path.append(pathname + os.sep + os.pardir + os.sep + "src")


# %% use class based approach

# define data path
path_to_data = (
    pathname + os.sep + os.pardir + os.sep + "CaloCem" + os.sep + "DATA"
)

# experiments via class
tam = tacalorimetry.Measurement(
    folder=path_to_data,
    regex=r"(myexp.*)",
    show_info=True,
    auto_clean=False,
)

# get sample and information
data = tam.get_data()
info = tam.get_information()

# add metadata
tam.add_metadata_source(
    pathlib.Path(pathname).parent / "tests" / "mini_metadata.csv", "experiment_nr"
)

# get meta
meta, meta_id = tam.get_metadata()

print(tam.get_metadata_grouping_options())


# %% average and std by group

import pandas as pd
import numpy as np

# vars
_groupby = "cement_name"
# _groupby = ["cement_name", "cement_amount_g"]
# _groupby = ["cement_amount_g", "cement_name"]
# _groupby = "cement_amount_g"
# _groupby = "water_amount_g"
# _groupby = ["date", "cement_name"]
_meta_id = "experiment_nr"
_meta_id_data = "sample_short"
_bin_width_s = 60  # [s]
_time_map = "left"  # "mid" "right"

# copy data
df = data.copy()

for value, group in meta.groupby(_groupby):
    print(value)

    # if one grouping level is used
    if isinstance(value, str) or isinstance(value, int):
        # info
        print(f"{_groupby:<20}: {value}")
        # modify data --> replace "sample_short" with metadata group name
        _idx_to_replace = df[_meta_id_data].isin(group[_meta_id])
        df.loc[_idx_to_replace, _meta_id_data] = str(value)
    # if multiple grouping levels are used
    elif isinstance(value, tuple):
        # info
        for _g, _v in zip(_groupby, value):
            # info
            print(f"{_g:<20}: {_v}")
        # modify data --> replace "sample_short" with metadata group name
        _idx_to_replace = df[_meta_id_data].isin(group[_meta_id])
        df.loc[_idx_to_replace, _meta_id_data] = " | ".join([str(x) for x in value])
    else:
        pass


# binning
df["BIN"] = pd.cut(df["time_s"], np.arange(0, 2 * 24 * 60 * 60, _bin_width_s))


# calculate average and std
df = (
    df.groupby([_meta_id_data, "BIN"])
    .agg(
        {
            "normalized_heat_flow_w_g": ["mean", "std"],
            "normalized_heat_j_g": ["mean", "std"],
        }
    )
    .dropna(thresh=2)
    .reset_index()
)

# regain "time_s" columns
df["time_s"] = [i.left for i in df["BIN"]]

# copy
df["sample"] = df[_meta_id_data]

# %%

# overwrite
tam._data = df

# plot
ax = tam.plot()
ax.set_ylim(0, 4)
