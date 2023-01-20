from TAInstCalorimetry import tacalorimetry
import pytest
import pysnooper
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
        # ("Exp_1.xls", 253680),
        # ("Exp_2.xls", 182040),
        # ("Exp_3.xls", 182040),
        # ("Exp_4.xls", 181320),
        # ("Exp_5.xls", 89760),
        # ("Exp_6.xls", 254160),
        # ("Exp_7.xls", 254160),
        # ("Exp_8.xls", 252000),
        # ("Mix_01.xls", 235470),
        # ("Mix_02.xls", 589200),
        # ("Mix_03.xls", 258600),
        # ("Mix_04.xls", 73920),
        # ("Mix_05.xls", 73920),
        # ("Mix_06.xls", 85080),
        # ("Mix_07.xls", 9960),
        # ("Mix_08.xls", 336480),
        ("opc_3.csv", 322461),
        # ("c3a.csv", 173929),
        # ("OPC_1.xls", 499440),
        ("OPC_2.xls", 336000),
    ],
)
@pysnooper.snoop()
def test_last_time_entry(test_input, expected):

    # path to data
    path_to_data = "DATA"
    # path_to_data = os.getcwd() + os.sep + os.pardir + os.sep + "DATA"

    # experiments via class
    tam = tacalorimetry.Measurement(folder=path_to_data, show_info=False)

    # get all data
    data = tam.get_data()

    # get "last time for a file
    last_time = int(
        data[data["sample"] == path_to_data + os.sep + test_input].tail(1)["time_s"]
    )

    # actual test
    assert last_time == int(expected)

# test_last_time_entry("opc_3.csv", 322461)
# test_last_time_entry("OPC_2.xls", 336000)
