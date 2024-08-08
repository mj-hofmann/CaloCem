import pathlib

from CaloCem import tacalorimetry


# path
path = pathlib.Path(__file__).parent.parent 
datapath = path / "CaloCem" / "DATA"
metadatapath = path / "tests"

# get data
tam = tacalorimetry.Measurement(
    datapath, regex="calorimetry_data_[1-5].*", show_info=True, cold_start=True, auto_clean=False
)
# %%
data = tam.get_data()

# add metadata
tam.add_metadata_source(metadatapath / "mini_metadata.csv", "experiment_nr")

# get meta
meta, meta_id = tam.get_metadata()


# %% plot by category

# # define action by two or more categories
categorize_by = ["cement_name", "date"]

# changes data!!!
tam.average_by_metadata(categorize_by)

# plot
ax = tam.plot(y_unit_milli=False)
ax.set_xlim(0,1)


# %% revert aggregation

# undo aggregation
tam.undo_average_by_metadata()


# %% plot de-aggregated data

# plot
ax = tam.plot()
ax.set_xlim(0, 1.5)

# %%
