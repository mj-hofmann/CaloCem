import pathlib

import pytest

from CaloCem import tacalorimetry


@pytest.mark.parametrize(
    "file, expected",
    [
        ("c3a.csv", 173873),
        (
            "gen1_calofile.csv",
            None,
        ),  # multiples samples in one file?  --> do nothing / return 0
        ("gen3_calofile.csv", 172799),
        ("calorimetry_data_1.csv", 416006),  
        ("calorimetry_data_2.csv", 277199),
        ("calorimetry_data_3.csv", 410349),  
        ("calorimetry_data_4.csv", 277199),
        ("calorimetry_data_5.csv", 277199),
        ("gen1_calofile3.csv", 264470),  # in-situ first gen file
        ("gen1_calofile2.csv", 263750),  # in-situ first gen file
        ("corrupt_example.csv", 259040),  # comma sep ("corrupt" due to 2 missing values)
    ],
)
def test_last_time_entry(file, expected):

    # path
    path = pathlib.Path().cwd() / "CaloCem" / "DATA"

    # get data
    data = tacalorimetry.Measurement(auto_clean=False)._read_calo_data_csv(path / file)

    # checks
    if data is None:
        # test
        assert data is None
    else:
        # discard NaN values
        data = data.dropna()
        # actual test
        assert int(data.tail(1)["time_s"].values[0]) == expected
