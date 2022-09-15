from TAInstCalorimetry import tacalorimetry
import pytest
import os

#
# run "time check" on each sample file
# see https://www.youtube.com/watch?v=DhUpxWjOhME (~ at 12:20)
# to use the the test:
#       1) install the module locally with "pip install -e ."
#       2) run "pytest" from shell
#
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("c3a.csv", 173929),
        ("OPC_1.xls", 499440),
        ("OPC_2.xls", 336000),
    ],
)
def test_last_time_entry(test_input, expected):

    # path to data
    path_to_data = os.getcwd() + os.sep + "DATA"

    # experiments via class
    tam = tacalorimetry.Measurement(folder=path_to_data, show_info=False)

    # get all data
    data = tam.get_data()

    # get "last time for a file
    last_time = int(
        data[data["sample"] == path_to_data + os.sep + test_input].tail(1)["time"]
    )

    # actual test
    assert last_time == expected
