import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from calocem.measurement import Measurement  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
METADATA = pathlib.Path(__file__).parent / "mini_metadata.csv"


@pytest.fixture
def tam():
    m = Measurement(DATA_DIR, regex=r"calorimetry_data_[1-4]\.csv", show_info=False)
    m.add_metadata_source(METADATA, "experiment_nr", show_info=False)
    return m


def test_plot_by_single_category_yields_one_plot_per_value(tam):
    expected = sorted(tam.get_metadata()[0]["cement_name"].unique())

    values = []
    for value, ax in tam.plot_by_category("cement_name"):
        values.append(value)
        assert value in ax.get_title()
        assert ax.get_lines()
        plt.close(ax.figure)

    assert sorted(values) == expected


def test_plot_by_multiple_categories_yields_tuples(tam):
    plots = list(tam.plot_by_category(["date", "cement_name"]))
    # calorimetry_data_5 is in the metadata but not loaded, so its group is skipped
    assert len(plots) == 2
    for value, ax in plots:
        assert isinstance(value, tuple) and len(value) == 2
        assert ax.get_lines()
        plt.close(ax.figure)


def test_plot_by_category_without_metadata_raises():
    m = Measurement(DATA_DIR, regex=r"calorimetry_data_1\.csv", show_info=False)
    with pytest.raises(ValueError, match="metadata"):
        next(m.plot_by_category("cement_name"))
