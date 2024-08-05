import os
import pathlib
import sys

from CaloCem import tacalorimetry

# get path of script and set it as current path
pathname = os.path.dirname(sys.argv[0])


# %% use class based approach

# define data path
path_to_data = (
    pathname + os.sep + os.pardir + os.sep + "CaloCem" + os.sep + "DATA"
)

# experiments via class
tam = tacalorimetry.Measurement(
    folder=path_to_data,
    regex=r"(myexp[5-8].*)",
    show_info=True,
    auto_clean=False,
)

# get sample and information
data = tam.get_data()


# %%

individual = True

peaks = tam.get_peaks(cutoff_min=15)

# sort
peaks = peaks.sort_values(by="normalized_heat_flow_w_g", ascending=True)

peaks = peaks.groupby(by="sample").last()  # .reset_index()

print(peaks)

# init empty list
astm_times = []

# loop samples
for sample, sample_data in tam._iter_samples():

    # pick sample data
    helper = data[data["sample"] == sample]

    # restrict to times before the peak
    helper = helper[helper["time_s"] <= peaks.at[sample, "time_s"]]

    # restrict to relevant heatflows the peak
    if individual == True:
        helper = helper[
            helper["normalized_heat_flow_w_g"]
            <= peaks.at[sample, "normalized_heat_flow_w_g"] * 0.50
        ]
    else:
        # use half-maximum average
        helper = helper[
            helper["normalized_heat_flow_w_g"]
            <= peaks["normalized_heat_flow_w_g"].mean() * 0.50
        ]

    # # add to list of of selected points
    astm_times.append(helper.tail(1))

# build overall DataFrame
astm_times = tacalorimetry.pd.concat(astm_times)
