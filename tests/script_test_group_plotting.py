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


# %% plot by category

categorize_by = (
    "cement_name"  # 'date', 'cement_name', 'cement_amount_g', 'water_amount_g'
)

# loop through plots via generator
for this_plot in tam.plot_by_category(
    categorize_by,
):
    # extract parts obtained from generator
    category_value, ax = this_plot

    # fine tuning of plot/cosmetics
    ax.set_title(f'My samples for "{category_value}" in category {categorize_by}')
    ax.set_ylim(0, 3)

    tacalorimetry.plt.show()
