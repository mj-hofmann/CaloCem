import pathlib

from TAInstCalorimetry import tacalorimetry


# path
path = pathlib.Path().cwd().parent / "TAInstCalorimetry" / "DATA"

# get data
tam = tacalorimetry.Measurement(
    path, regex="myexp.*", show_info=True, cold_start=False, auto_clean=False
)
# %%
data = tam.get_data()

# add metadata
tam.add_metadata_source("mini_metadata.csv", "experiment_nr")

# get meta
meta, meta_id = tam.get_metadata()
