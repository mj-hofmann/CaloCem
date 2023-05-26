import pathlib

from TAInstCalorimetry import tacalorimetry


# path
path = pathlib.Path().cwd().parent / "TAInstCalorimetry" / "DATA"

# get data
tam = tacalorimetry.Measurement(
    path, regex="myexp.*", show_info=True, cold_start=True, auto_clean=False
)
# %%
data = tam.get_data()

# add metadata
tam.add_metadata_source("mini_metadata.csv", "experiment_nr")

# get meta
meta, meta_id = tam.get_metadata()


# %% plot by category

# define action by one category
categorize_by = "cement_name"  # 'date', 'cement_amount_g', 'water_amount_g'

# # define action by two or more categories
categorize_by = ["date", "cement_name"]

# changes data!!!
tam.average_by_metadata(categorize_by)

# plot
tam.plot()


# %% revert aggregation

# undo aggregation
tam.undo_average_by_metadata()


# %% plot de-aggregated data

# plot
tam.plot()
