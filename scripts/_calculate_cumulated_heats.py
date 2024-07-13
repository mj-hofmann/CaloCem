import pathlib

from TAInstCalorimetry import tacalorimetry

# %% use class based approach

# define data path
path_to_data = pathlib.Path().cwd().parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = tacalorimetry.Measurement(
    folder=path_to_data.as_posix(),
    regex=r"(.*data_[12345].csv$)",
    # regex=r"(.*xls$)",
    show_info=True,
    auto_clean=False,
)

# get sample and information
data = tam.get_data()

# %% plot

# tam.plot(y="heat_j")

# list of dfs
list_of_dfs = []

# % loop
for sample, roi in tam._iter_samples():
    print(sample)

    try:
        if not roi["heat_j"].isna().all():
            # use as is
            list_of_dfs.append(roi)
            # go to next
            continue
    except KeyError as e:
        # info
        print(e)

    # info
    print(f'==> Inferring "heat_j" column for {sample}')

    # get target rows
    roi = roi.dropna(subset=["heat_flow_w"]).sort_values(by="time_s")

    # inferring cumulated heat using the "trapezoidal integration method"

    # introduce helpers
    roi["_h1_y"] = 0.5 * (roi["heat_flow_w"] + roi["heat_flow_w"].shift(1)).shift(-1)
    roi["_h2_x"] = (roi["time_s"] - roi["time_s"].shift(1)).shift(-1)

    # integrate
    roi["heat_j"] = (roi["_h1_y"] * roi["_h2_x"]).cumsum()

    # clean
    del roi["_h1_y"], roi["_h2_x"]

    # append to list
    list_of_dfs.append(roi)

# set data
tam._data = tacalorimetry.pd.concat(list_of_dfs)


# get sample and information
data = tam.get_data()

# plot
tam.plot(
    y="heat_j"
    # y="heat_flow_w"
)
