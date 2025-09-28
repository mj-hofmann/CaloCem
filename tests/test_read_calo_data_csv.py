import pathlib

import pytest

from calocem.measurement import Measurement


@pytest.mark.parametrize(
    "file, expected",
    [
        ("c3a.csv", 173852),
        (
            "gen1_calofile.csv",
            0,
        ),  # multiples samples in one file?  --> do nothing / return 0
        ("gen3_calofile.csv", 172769),
        ("calorimetry_data_1.csv", 416006),  
        ("calorimetry_data_2.csv", 277199),
        ("calorimetry_data_3.csv", 410349),  
        ("calorimetry_data_4.csv", 277199),
       # ("calorimetry_data_5.csv", 277199),
        ("gen1_calofile3.csv", 264470),  # in-situ first gen file
        ("gen1_calofile2.csv", 263750),  # in-situ first gen file
        ("corrupt_example.csv", 259006),  # comma sep ("corrupt" due to 2 missing values)
    ],
)
def test_last_time_entry(file, expected):

    # path
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"

    # get data
    tam = Measurement(path, regex=file)
    data = tam.get_data()
    # checks
    if data.empty:
        # test
        assert len(data) == 0
    else:
        # discard NaN values
        data = data.dropna()
        # actual test
        assert int(data.tail(1)["time_s"].values[0]) == expected
