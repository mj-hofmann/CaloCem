import csv
import logging
import os
import pathlib
import pickle
import re
import warnings
from dataclasses import dataclass, field

import matplotlib
import matplotlib.axes
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysnooper
from scipy import integrate, signal
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter

from calocem import utils

from .processparams import *

logging.basicConfig(
    filename="CaloCem.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    filemode="a",
    level=logging.INFO,
)


class AutoCleanException(Exception):
    def __init__(self):
        message = "auto_clean failed. Consider switching to turn this option off."
        super().__init__(message)


class ColdStartException(Exception):
    def __init__(self):
        message = "cold_start failed. Consider switching to cold_start=True."
        super().__init__(message)


class AddMetaDataSourceException(Exception):
    def __init__(self, list_of_possible_ids):
        message = "The specified id column is not available in the declared file. Please use one of"
        for option in list_of_possible_ids:
            # extend string
            message += f"\n  - {option}"
        # show message
        super().__init__(message)


#
# Base class of "ta-calorimetry"
#
class Measurement:
    """
    A base class for handling and processing isothermal heat flow calorimetry data.

    Currently supported file formats are .xls and .csv files.
    Only TA Instruments data files are supported at the moment.

    Parameters
    ----------
    folder : str, optional
        path to folder containing .xls and/or .csv experimental result
        files. The default is None.
    show_info : bool, optional
        whether or not to print some informative lines during code
        execution. The default is True.
    regex : str, optional
        regex pattern to include only certain experimental result files
        during initialization. The default is None.
    auto_clean : bool, optional
        whether or not to exclude NaN values contained in the original
        files and combine data from differently names temperature columns.
        The default is False.
    cold_start : bool, optional
        whether or not to use "pickled" files for initialization; save time
        on reading

    Examples
    --------

    >>> import CaloCem as ta
    >>> from pathlib import Path
    >>>
    >>> calodatapath = Path(__file__).parent
    >>> tam = ta.Measurement(folder=calodatapath, show_info=True)

    We can use a regex pattern to only include certain files in the datafolder. Here we assume that we only want to load .csv files which contain the string "bm".

    >>> tam = ta.Measurement(folder=calodatapath, regex=r".*bm.*.csv", show_info=True)

    """

    # init
    _info = pd.DataFrame()
    _data = pd.DataFrame()
    _data_unprocessed = (
        pd.DataFrame()
    )  # helper to store experimental data as loaded from files

    # further metadata
    _metadata = pd.DataFrame()
    _metadata_id = ""

    # define pickle filenames
    _file_data_pickle = pathlib.Path().cwd() / "_data.pickle"
    _file_info_pickle = pathlib.Path().cwd() / "_info.pickle"

    #
    # init
    #
    def __init__(
        self,
        folder=None,
        show_info=True,
        regex=None,
        auto_clean=False,
        cold_start=True,
        processparams=None,
        new_code=False,
        processed=False,
    ):
        """
        intialize measurements from folder


        """
        self._new_code = new_code
        self._processed = processed

        if not isinstance(processparams, ProcessingParameters):
            self.processparams = ProcessingParameters()
        else:
            self.processparams = processparams

        # read
        if folder:
            if cold_start:
                # get data and parameters
                self._get_data_and_parameters_from_folder(
                    folder, regex=regex, show_info=show_info
                )
            else:
                # get data and parameters from pickled files
                self._get_data_and_parameters_from_pickle()
            try:
                if auto_clean:
                    # remove NaN values and merge time columns
                    self._auto_clean_data()
            except Exception as e:
                # info
                print(e)
                raise AutoCleanException
                # return
                return

        if self.processparams.downsample.apply:
            self._apply_adaptive_downsampling()
        # Message
        print(
            "================\nAre you missing some samples? Try rerunning with auto_clean=True and cold_start=True.\n================="
        )

    #
    # get_data_and_parameters_from_folder
    #
    def _get_data_and_parameters_from_folder(self, folder, regex=None, show_info=True):
        """
        get_data_and_parameters_from_folder
        """

        if not isinstance(folder, str):
            # convert
            folder = str(folder)

        # loop folder
        for f in os.listdir(folder):
            if not f.endswith((".xls", ".csv")):
                # go to next
                continue

            if regex:
                # check match
                if not re.match(regex, f):
                    # skip this file
                    continue

            # info
            if show_info:
                print(f"Reading {f}.")

            # define file
            file = folder + os.sep + f

            # check xls
            if f.endswith(".xls"):
                if self._new_code:
                    self._data = pd.concat(
                        [
                            self._data,
                            self._read_csv_data(file, show_info=show_info),
                        ]
                    )
                if self._new_code is False:
                    # collect information
                    try:
                        self._info = pd.concat(
                            [
                                self._info,
                                self._read_calo_info_xls(file, show_info=show_info),
                            ]
                        )
                    except Exception:
                        # initialize
                        if self._info.empty:
                            self._info = self._read_calo_info_xls(
                                file, show_info=show_info
                            )

                    # collect data
                    try:
                        self._data = pd.concat(
                            [
                                self._data,
                                self._read_calo_data_xls(file, show_info=show_info),
                            ]
                        )

                    except Exception:
                        # initialize
                        if self._data.empty:
                            self._data = self._read_calo_data_xls(
                                file, show_info=show_info
                            )

            # append csv
            if f.endswith(".csv"):
                # collect data
                if self._new_code:
                    self._data = pd.concat(
                        [
                            self._data,
                            self._read_csv_data(file, show_info=show_info),
                        ]
                    )
                    # self._read_csv_data(file, show_info=show_info)
                if self._new_code is False:
                    try:
                        self._data = pd.concat(
                            [
                                self._data,
                                self._read_calo_data_csv(file, show_info=show_info),
                            ]
                        )

                    except Exception:
                        # initialize
                        if self._data.empty:
                            self._data = self._read_calo_data_csv(
                                file, show_info=show_info
                            )

                # collect information
                try:
                    self._info = pd.concat(
                        [
                            self._info,
                            self._read_calo_info_csv(file, show_info=show_info),
                        ]
                    )
                except Exception:
                    # initialize
                    if self._info.empty:
                        try:
                            self._info = self._read_calo_info_csv(
                                file, show_info=show_info
                            )
                        except Exception:
                            pass

        # get "heat_j" columns if the column is not part of the source files
        if self.processparams.preprocess.infer_heat:
            try:
                self._infer_heat_j_column()
            except Exception:
                pass

        # if self.processparams.downsample.apply is not False:
        #     self._apply_adaptive_downsampling()
        # write _data and _info to pickle
        with open(self._file_data_pickle, "wb") as f:
            pickle.dump(self._data, f)
        with open(self._file_info_pickle, "wb") as f:
            pickle.dump(self._info, f)

        # store experimental data for recreating state after reading from files
        self._data_unprocessed = self._data.copy()

    #
    # get data and information from pickled files
    #
    def _get_data_and_parameters_from_pickle(self):
        """
        get data and information from pickled files

        Returns
        -------
        None.

        """

        # read from pickle
        try:
            self._data = pd.read_pickle(self._file_data_pickle)
            self._info = pd.read_pickle(self._file_info_pickle)
            # store experimental data for recreating state after reading from files
            self._data_unprocessed = self._data.copy()
        except FileNotFoundError:
            # raise custom Exception
            raise ColdStartException()

        # log
        logging.info("_data and _info loaded from pickle files.")

    #
    # determine csv data range
    #
    def _determine_data_range_csv(self, file):
        """
        determine csv data range of CSV-file.

        Parameters
        ----------
        file : str
            filepath.

        Returns
        -------
        empty_lines : TYPE
            DESCRIPTION.

        """
        # open csv file
        thefile = open(file)
        # detect empty lines which are characteristic at the beginning and
        # end of the data block
        empty_lines = [
            index for index, line in enumerate(csv.reader(thefile)) if len(line) == 0
        ]
        return empty_lines

    def _read_csv_data(self, file, show_info=True):
        """
        NEW IMPLEMENTATION
        """
        filetype = pathlib.Path(file).suffix
        if not self._processed:
            if filetype == ".csv":
                delimiter = utils.detect_delimiter(file)
                title_row = utils.find_title_row(file, delimiter)
            else:
                delimiter = None
                title_row = 0

            data = utils.load_data(file, delimiter, title_row)

            start_time = utils.find_reaction_start_time(data)

            if delimiter == "\t":
                data = utils.prepare_tab_columns(data, file)
            else:
                if filetype == ".csv":
                    data = utils.tidy_colnames(data)

            data = utils.remove_unnecessary_data(data)
            data = utils.convert_df_to_float(data)
            data = utils.correct_start_time(data, start_time)
            data = utils.add_sample_info(data, file)

        elif self._processed:
            data = pd.read_csv(file, sep=",", header=0)

        return data

    #
    # read csv data
    #
    def _read_calo_data_csv(self, file, show_info=True):
        """
        try reading calorimetry data from csv file via multiple options

        Parameters
        ----------
        file : str | pathlib.Path
            path to csv fileto be read.
        show_info : bool, optional
            flag whether or not to show information. The default is True.

        Returns
        -------
        pd.DataFrame

        """

        try:
            data = self._read_calo_data_csv_comma_sep(file, show_info=show_info)
        except Exception:
            data = self._read_calo_data_csv_tab_sep(file, show_info=show_info)

        # valid read
        if data is None:
            # log
            logging.info(f"\u2716 reading {file} FAILED.")

        # log
        logging.info(f"\u2714 reading {file} successful.")

        # return
        return data

    #
    # read csv data
    #
    def _read_calo_data_csv_comma_sep(self, file, show_info=True):
        """
        read data from csv file

        Parameters
        ----------
        file : str
            filepath.

        Returns
        -------
        data : pd.DataFrame
            experimental data contained in file.

        """

        # define Excel file
        data = pd.read_csv(
            file, header=None, sep="No meaningful separator", engine="python"
        )

        # check for tab-separation
        if "\t" in data.at[0, 0]:
            # raise Exception
            raise ValueError

        # look for potential index indicating in-situ-file
        if data[0].str.contains("Reaction start").any():
            # get target row
            helper = data[0].str.contains("Reaction start")
            # get row
            start_row = helper[helper].index.tolist()[0]
            # get offset for in-situ files
            t_offset_in_situ_s = float(data.at[start_row, 0].split(",")[0])

        data = utils.parse_rowwise_data(data)
        data = utils.tidy_colnames(data)

        data = utils.remove_unnecessary_data(data)

        # type conversion
        data = utils.convert_df_to_float(data)

        # check for "in-situ" sample --> reset
        try:
            # offset
            data["time_s"] -= t_offset_in_situ_s
            # write to log
            logging.info(
                f"\u26a0 Consider {file} as in-situ-file --> time-scale adjusted."
            )
        except Exception:
            pass

        # restrict to "time_s" > 0
        data = data.query("time_s > 0").reset_index(drop=True)

        # add sample information
        data = utils.add_sample_info(data, file)

        # if self.processparams.downsample.apply:
        #     data = self._apply_adaptive_downsampling(data)

        # return
        return data

    #
    # read csv data
    #
    def _read_calo_data_csv_tab_sep(self, file: str, show_info=True) -> pd.DataFrame:
        """
        Parameters
        ----------
        file : str | pathlib.Path
            path to tab separated csv-files from "older" versions of the device.
        show_info : bool, optional
            flag whether or not to show information. The default is True.

        Returns
        -------
        pd.DataFrame

        """

        # read
        raw = pd.read_csv(file, sep="\t", header=None)

        # process
        data = raw.copy()

        # get sample mass (if available)
        try:
            # get mass, first row in 3rd column is the title
            # the assumption is that the sample weight is one value on top of the 3rd column
            mass_index = raw.index[raw.iloc[:, 3].notna()]
            mass = float(raw.iloc[mass_index[1], 3])
        except IndexError:
            # set mass to None
            mass = None
            # go on
            pass

        # get "reaction start" time (if available)
        try:
            # get "reaction start" time in seconds
            _helper = data[data.iloc[:, 2].str.lower() == "reaction start"].head(1)
            # convert to float
            t0 = float(_helper[0].values[0])
        except Exception:
            # set t0 to None
            t0 = None
            # go on
            pass

        # remove all-Nan columns
        data = data.dropna(how="all", axis=1)

        # restrict to first two columns
        data = data.iloc[:, :2]

        # rename
        try:
            data.columns = ["time_s", "heat_flow_mw"]
        except ValueError:
            # return empty DataFrame
            return pd.DataFrame({"time_s": 0}, index=[0])

        # get data columns
        data = data.loc[3:, :].reset_index(drop=True)

        # convert data types
        data["time_s"] = data["time_s"].astype(float)
        data["heat_flow_mw"] = data["heat_flow_mw"].apply(
            lambda x: float(x.replace(",", "."))
        )

        # convert to same unit
        data["heat_flow_w"] = data["heat_flow_mw"] / 1000

        # calculate cumulative heat flow
        data["heat_j"] = integrate.cumulative_trapezoid(
            data["heat_flow_w"], x=data["time_s"], initial=0
        )

        # remove "heat_flow_w" column
        del data["heat_flow_mw"]

        # take into account time offset via "reactin start" time
        if t0:
            data["time_s"] -= t0

        # calculate normalized heat flow and heat
        if mass:
            data["normalized_heat_flow_w_g"] = data["heat_flow_w"] / mass
            data["normalized_heat_j_g"] = data["heat_j"] / mass

        # restrict to "time_s" > 0
        data = data.query("time_s >= 0").reset_index(drop=True)

        # add sample information
        data["sample"] = file
        data["sample_short"] = pathlib.Path(file).stem

        # type conversion
        data = utils.convert_df_to_float(data)

        # return
        return data

    #
    # read csv info
    #
    def _read_calo_info_csv(self, file, show_info=True):
        """
        read info from csv file

        Parameters
        ----------
        file : str
            filepath.

        Returns
        -------
        info : pd.DataFrame
            information (metadata) contained in file

        """

        try:
            # determine number of lines to skip
            empty_lines = self._determine_data_range_csv(file)
            # read info block from csv-file
            info = pd.read_csv(
                file, nrows=empty_lines[0] - 1, names=["parameter", "value"]
            ).dropna(subset=["parameter"])
            # the last block is not really meta data but summary data and
            # somewhat not necessary
        except IndexError:
            # return empty DataFrame
            info = pd.DataFrame()

        # add sample name as column
        info["sample"] = file
        info["sample_short"] = pathlib.Path(file).stem

        # return
        return info

    #
    # read excel info
    #
    def _read_calo_info_xls(self, file, show_info=True):
        """
        read information from xls-file

        Parameters
        ----------
        file : str
            filepath.
        show_info : bool, optional
            flag whether or not to show information. The default is True.

        Returns
        -------
        info : pd.DataFrame
            information (metadata) contained in file

        """
        # specify Excel
        xl = pd.ExcelFile(file)

        try:
            # get experiment info (first sheet)
            df_experiment_info = xl.parse(
                sheet_name="Experiment info", header=0, names=["parameter", "value"]
            ).dropna(subset=["parameter"])
            # use first row as header
            df_experiment_info = df_experiment_info.iloc[1:, :]

            # add sample information
            df_experiment_info["sample"] = file
            df_experiment_info["sample_short"] = pathlib.Path(file).stem

            # rename variable
            info = df_experiment_info

            # return
            return info

        except Exception as e:
            if show_info:
                print(e)
                print(f"==> ERROR in file {file}")

    #
    # read excel data
    #
    def _read_calo_data_xls(self, file, show_info=True):
        """
        read data from xls-file

        Parameters
        ----------
        file : str
            filepath.
        show_info : bool, optional
            flag whether or not to show information. The default is True.

        Returns
        -------
        data : pd.DataFrame
            data contained in file

        """

        # define Excel file
        xl = pd.ExcelFile(file)

        try:
            # parse "data" sheet
            df_data = xl.parse("Raw data", header=None)

            # replace init timestamp
            df_data.iloc[0, 0] = "time"

            # get new column names
            new_columnames = []
            for i, j in zip(df_data.iloc[0, :], df_data.iloc[1, :]):
                # build
                new_columnames.append(
                    re.sub(r"[\s\n\[\]\(\)° _]+", "_", f"{i}_{j}".lower())
                    .replace("/", "_")
                    .replace("_signal_", "_")
                )

            # set
            df_data.columns = new_columnames

            # cut out data part
            df_data = df_data.iloc[2:, :].reset_index(drop=True)

            # drop column
            try:
                df_data = df_data.drop(columns=["time_markers_nan"])
            except KeyError:
                pass

            # remove columns with too many NaNs
            df_data = df_data.dropna(axis=1, thresh=3)
            # # remove rows with NaNs
            # df_data = df_data.dropna(axis=0)

            # float conversion
            for _c in df_data.columns:
                # convert
                df_data[_c] = df_data[_c].astype(float)

            # add sample information
            df_data["sample"] = file
            df_data["sample_short"] = pathlib.Path(file).stem

            # rename
            data = df_data

            # log
            logging.info(f"\u2714 reading {file} successful.")

            # return
            return data

        except Exception as e:
            if show_info:
                print(
                    "\n\n==============================================================="
                )
                print(f"{e} in file '{pathlib.Path(file).name}'")
                print("Please, rename the data sheet to 'Raw data' (device default).")
                print(
                    "===============================================================\n\n"
                )

            # log
            logging.info(f"\u2716 reading {file} FAILED.")

            # return
            return None

    #
    # iterate samples
    #
    def _iter_samples(self, regex=None):
        """
        iterate samples and return corresponding data

        Returns
        -------
        sample (str) : name of the current sample
        data (pd.DataFrame) : data corresponding to the current sample
        """

        for sample, data in self._data.groupby(by="sample"):
            if regex:
                if not re.findall(regex, sample):
                    continue

            yield sample, data

    #
    # auto clean data
    #
    def _auto_clean_data(self):
        """
        remove NaN values from self._data and merge differently named columns
        representing the (constant) temperature set for the measurement

        Returns
        -------
        None.

        """

        # remove NaN values and reset index
        self._data = self._data.dropna(
            subset=[c for c in self._data.columns if re.match("normalized_heat", c)]
        ).reset_index(drop=True)

        # determine NaN count
        nan_count = self._data["temperature_temperature_c"].isna().astype(
            int
        ) + self._data["temperature_c"].isna().astype(int)

        # consolidate temperature columns
        if (
            "temperature_temperature_c" in self._data.columns
            and "temperature_c" in self._data.columns
        ):
            # use values from column "temperature_c" and set the values to column
            # "temperature_c"
            self._data.loc[
                (self._data["temperature_temperature_c"].isna()) & (nan_count == 1),
                "temperature_temperature_c",
            ] = self._data.loc[
                (~self._data["temperature_c"].isna()) & (nan_count == 1),
                "temperature_c",
            ]

            # remove values from column "temperature_c"
            self._data = self._data.drop(columns=["temperature_c"])

        # rename column
        self._data = self._data.rename(
            columns={"temperature_temperature_c": "temperature_c"}
        )

    #
    # plot
    #
    def plot(
        self,
        t_unit="h",
        y="normalized_heat_flow_w_g",
        y_unit_milli=True,
        regex=None,
        show_info=True,
        ax=None,
    ):
        """

        Plot the calorimetry data.

        Parameters
        ----------
        t_unit : str, optional
            time unit. The default is "h". Options are "s", "min", "h", "d".
        y : str, optional
            y-axis. The default is "normalized_heat_flow_w_g". Options are
            "normalized_heat_flow_w_g", "heat_flow_w", "normalized_heat_j_g",
            "heat_j".
        y_unit_milli : bool, optional
            whether or not to plot y-axis in Milliwatt. The default is True.
        regex : str, optional
            regex pattern to include only certain samples during plotting. The
            default is None.
        show_info : bool, optional
            whether or not to show information. The default is True.
        ax : matplotlib.axes._axes.Axes, optional
            axis to plot to. The default is None.

        Examples
        --------
        >>> import CaloCem as ta
        >>> from pathlib import Path
        >>>
        >>> calodatapath = Path(__file__).parent
        >>> tam = ta.Measurement(folder=calodatapath, show_info=True)
        >>> tam.plot(t_unit="h", y="normalized_heat_flow_w_g", y_unit_milli=False)

        """

        # y-value
        if y == "normalized_heat_flow_w_g":
            y_column = "normalized_heat_flow_w_g"
            y_label = "Normalized Heat Flow / [W/g]"
        elif y == "heat_flow_w":
            y_column = "heat_flow_w"
            y_label = "Heat Flow / [W]"
        elif y == "normalized_heat_j_g":
            y_column = "normalized_heat_j_g"
            y_label = "Normalized Heat / [J/g]"
        elif y == "heat_j":
            y_column = "heat_j"
            y_label = "Heat / [J]"

        if y_unit_milli:
            y_label = y_label.replace("[", "[m")

        # x-unit
        if t_unit == "s":
            x_factor = 1.0
        elif t_unit == "min":
            x_factor = 1 / 60
        elif t_unit == "h":
            x_factor = 1 / (60 * 60)
        elif t_unit == "d":
            x_factor = 1 / (60 * 60 * 24)

        # y-unit
        if y_unit_milli:
            y_factor = 1000
        else:
            y_factor = 1

        for sample, data in self._iter_samples():
            if regex:
                if not re.findall(rf"{regex}", os.path.basename(sample)):
                    continue
            data["time_s"] = data["time_s"] * x_factor
            # all columns containing heat
            heatcols = [s for s in data.columns if "heat" in s]
            data[heatcols] = data[heatcols] * y_factor
            ax, _ = utils.create_base_plot(data, ax, "time_s", y_column, sample)
            ax = utils.style_base_plot(
                ax,
                y_label,
                t_unit,
                sample,
            )
        return ax

    #
    # plot by category
    #
    def plot_by_category(
        self, categories, t_unit="h", y="normalized_heat_flow_w_g", y_unit_milli=True
    ):
        """
        plot by category, wherein the category is based on the information passed
        via "self._add_metadata_source". Options available as "category" are
        accessible via "self.get_metadata_grouping_options"

        Parameters
        ----------
        categories : str, list[str]
            category (from "self.get_metadata_grouping_options") to group by.
            specify a string or a list of strings here
        t_unit : TYPE, optional
            see "self.plot". The default is "h".
        y : TYPE, optional
            see "self.plot". The default is "normalized_heat_flow_w_g".
        y_unit_milli : TYPE, optional
            see "self.plot". The default is True.

        Examples
        --------
        >>> import CaloCem as ta
        >>> from pathlib import Path
        >>>
        >>> calodatapath = Path(__file__).parent
        >>> tam = ta.Measurement(folder=calodatapath, show_info=True)
        >>> tam.plot_by_category(categories="sample")


        Returns
        -------
        None.

        """

        def build_helper_string(values: list) -> str:
            """
            build a "nicely" formatted string from a supplied list
            """

            if len(values) == 2:
                # connect with "and"
                formatted = " and ".join([str(i) for i in values])
            elif len(values) > 2:
                # connect with comma and "and" for last element
                formatted = (
                    ", ".join([str(i) for i in values[:-1]]) + " and " + str(values[-1])
                )
            else:
                formatted = "---"

            # return
            return formatted

        # loop category values
        for selections, _ in self._metadata.groupby(by=categories):
            if isinstance(selections, tuple):
                # - if multiple categories to group by are specified -
                # init helper DataFrame
                target_idx = pd.DataFrame()
                # identify corresponding samples
                for selection, category in zip(selections, categories):
                    target_idx[category] = self._metadata[category] == selection
                # get relevant indices
                target_idx = target_idx.sum(axis=1) == len(categories)
                # define title
                title = f"Grouped by {build_helper_string(categories)} ({build_helper_string(selections)})"
            else:
                # - if only one(!) category to group by is specified -
                # identify corresponding samples
                target_idx = self._metadata[categories] == selections
                # define title
                title = f"Grouped by {categories} ({selections})"

            # pick relevant samples
            target_samples = self._metadata.loc[target_idx, self._metadata_id]

            # build corresponding regex
            regex = "(" + ")|(".join(target_samples) + ")"

            # plot
            ax = self.plot(regex=regex, t_unit=t_unit, y=y, y_unit_milli=y_unit_milli)

            # set title
            ax.set_title(title)

            # yield latest plot
            yield selections, ax

    @staticmethod
    def _plot_peak_positions(
        data,
        ax,
        _age_col,
        _target_col,
        peaks,
        sample,
        plt_top,
        plt_right_s,
        plot_labels,
        xmarker,
    ):
        """
        Plot detected peaks.
        """

        ax, new_ax = utils.create_base_plot(data, ax, _age_col, _target_col, sample)

        if xmarker:
            ax.plot(
                data[_age_col][peaks],
                data[_target_col][peaks],
                "x",
                color="red",
            )

        ax.vlines(
            x=data[_age_col][peaks],
            ymin=0,
            ymax=data[_target_col][peaks],
            color="red",
        )

        if plot_labels:
            for x, y in zip(data[_age_col][peaks], data[_target_col][peaks]):
                y = y + 0.0002
                ax.text(x, y, f"{round(x, 2)}", color="red")

            # ax.text(
            #     x=data[_age_col][peaks],
            #     y=data[_target_col][peaks],
            #     #s=[f"{round(i,2)}" for i in data[_age_col][peaks]],
            #     s="hallo",
            #     color="red",
            # )

        limits = {
            "left": ax.get_xlim()[0],
            "right": plt_right_s,
            "bottom": 0,
            "top": plt_top,
        }

        ax = utils.style_base_plot(ax, _target_col, _age_col, sample, limits)

        if new_ax:
            plt.show()

    @staticmethod
    def _plot_maximum_slope(
        data,
        ax,
        age_col,
        target_col,
        sample,
        characteristics,
        time_discarded_s,
        save_path=None,
        xscale="log",
        xunit="s",
    ):
        x_increment = 600
        if xunit == "h":
            data[age_col] = data[age_col] / 3600
            characteristics[age_col] = characteristics[age_col] / 3600
            time_discarded_s = time_discarded_s / 3600
            x_increment = 0.2

        ax, new_ax = utils.create_base_plot(data, ax, age_col, target_col, sample)

        ax2 = ax.twinx()
        # plot gradient
        ax2.plot(data[age_col], data["gradient"], label="Gradient", color="orange")
        ax2.set_yscale("linear")
        xmask = data[age_col] > time_discarded_s
        y_vals = data[target_col][xmask]
        ymin = y_vals.min() + y_vals.min() * 0.1
        ymax = y_vals.max() + y_vals.max() * 0.1
        # ax.set_ylim(ymin, ymax)

        y2_vals = data["gradient"][xmask]
        y2min = y2_vals.min() + y2_vals.min() * 0.1
        y2max = y2_vals.max() + y2_vals.max() * 0.1
        ax2.set_ylim(y2min, y2max)

        ax2.set_ylabel(r"Gradient [Wg$^{-1}$s$^{-1}$]")

        # add vertical lines
        for _idx, _row in characteristics.iterrows():
            # vline
            t_maxslope = _row.at[age_col]
            ax.axvline(t_maxslope, color="green", alpha=0.3)

        if xunit == "h":
            limits = {"left": 0.1, "right": ax.get_xlim()[1], "bottom": 0, "top": ymax}

        else:
            limits = {"left": 100, "right": ax.get_xlim()[1], "bottom": 0, "top": ymax}

        ax = utils.style_base_plot(
            ax,
            target_col,
            age_col,
            sample,
            limits,
            time_discarded_s=time_discarded_s,
            xunit=xunit,
        )

        ax.set_xscale(xscale)
        ax.text(
            t_maxslope + x_increment,
            0.00025,
            f"{round(t_maxslope, 2)} {xunit} ",
            color="green",
        )

        if new_ax:
            if save_path:
                sample_name = pathlib.Path(sample).stem
                plt.savefig(save_path / f"maximum_slope_detect_{sample_name}.pdf")
            else:
                plt.show()

    @staticmethod
    def _plot_intersection(
        data,
        ax,
        age_col,
        target_col,
        sample,
        # characteristics,
        time_discarded_s,
        characteristics,
        save_path=None,
        xscale="log",
        # xunit="s",
        hmax=None,
        tmax=None,
    ):
        if characteristics.xunit == "h":
            data.loc[:, age_col] = data.loc[:, age_col] / 3600
            characteristics.time_s = characteristics.time_s / 3600
            characteristics.dorm_time_s = characteristics.dorm_time_s / 3600
            characteristics.gradient = characteristics.gradient * 3600
            tmax = tmax / 3600
            characteristics.x_intersect = characteristics.x_intersect / 3600
            # characteristics[age_col] = characteristics[age_col] / 3600

        ax, new_ax = utils.create_base_plot(data, ax, age_col, target_col, sample)
        # print(new_ax)
        ax = utils.style_base_plot(
            ax,
            target_col,
            age_col,
            sample,
            time_discarded_s=time_discarded_s,
            xunit=characteristics.xunit,
        )

        ax.axline(
            (characteristics.time_s, characteristics.normalized_heat_flow_w_g),
            slope=characteristics.gradient,
            color="red",
            linestyle="--",
        )
        if characteristics.intersection == "dormant_hf":
            ax.axhline(
                y=characteristics.dorm_normalized_heat_flow_w_g,
                color="red",
                linestyle="--",
            )
            ax.text(
                x=characteristics.x_intersect,
                y=characteristics.dorm_normalized_heat_flow_w_g,
                s=rf"   $t_i=$ {characteristics.x_intersect:.1f} {characteristics.xunit}"
                + "\n",
                color="green",
            )
        elif characteristics.intersection == "abscissa":
            ax.text(
                x=characteristics.x_intersect,
                y=0,
                s=rf"   $t_i=$ {characteristics.x_intersect:.1f} {characteristics.xunit}"
                + "\n",
                color="green",
            )

        ax.axvline(
            x=characteristics.x_intersect,
            color="green",
            linestyle=":",
        )

        ax.set_xscale(xscale)
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, hmax)

        if new_ax:
            if save_path:
                sample_name = pathlib.Path(sample).stem
                plt.savefig(save_path / f"intersection_detect_{sample_name}.pdf")
            else:
                plt.show()

    #
    # get the cumulated heat flow a at a certain age
    #

    def get_cumulated_heat_at_hours(self, processparams=None, target_h=4, **kwargs):
        """
        get the cumulated heat flow a at a certain age

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. The default is None. If None, the default
            parameters are used. The most important parameter is the cutoff time
            in minutes which describes the initial time period of the measurement
            which is not considered for the cumulated heat flow. It is defined in
            the ProcessingParameters class. The default value is 30 minutes.

        target_h : int | float
            end time in hourscv

        Returns
        -------
        A Pandas dataframe

        """
        if "cutoff_min" in kwargs:
            cutoff_min = kwargs["cutoff_min"]
            warnings.warn(
                "The cutoff_min parameter is deprecated. Please use the ProcessingParameters class instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            if not processparams:
                processparams = ProcessingParameters()
            cutoff_min = processparams.cutoff.cutoff_min

        def applicable(df, target_h=4, cutoff_min=None):
            # convert target time to seconds
            target_s = 3600 * target_h
            # helper
            _helper = df.query("time_s <= @target_s").tail(1)
            # get heat at target time
            hf_at_target = float(_helper["normalized_heat_j_g"].values[0])

            # if cutoff time specified
            if cutoff_min:
                # convert target time to seconds
                target_s = 60 * cutoff_min
                try:
                    # helper
                    _helper = df.query("time_s <= @target_s").tail(1)
                    # type conversion
                    hf_at_cutoff = float(_helper["normalized_heat_j_g"].values[0])
                    # correct heatflow for heatflow at cutoff
                    hf_at_target = hf_at_target - hf_at_cutoff
                except TypeError:
                    name_wt_nan = df["sample_short"].tolist()[0]
                    print(
                        f"Found NaN in Normalized heat of sample {name_wt_nan} searching for cumulated heat at {target_h}h and a cutoff of {cutoff_min}min."
                    )
                    return np.nan

            # return
            return hf_at_target

        # in case of one specified time
        if isinstance(target_h, int) or isinstance(target_h, float):
            # groupby
            results = (
                self._data.groupby(by="sample")[["time_s", "normalized_heat_j_g"]]
                .apply(
                    lambda x: applicable(x, target_h=target_h, cutoff_min=cutoff_min),
                )
                .reset_index(level=0)
            )
            # rename
            results.columns = ["sample", "cumulated_heat_at_hours"]
            results["target_h"] = target_h
            results["cutoff_min"] = cutoff_min

        # in case of specified list of times
        elif isinstance(target_h, list):
            # init list
            list_of_results = []
            # loop
            for this_target_h in target_h:
                # groupby
                _results = (
                    self._data.groupby(by="sample")[["time_s", "normalized_heat_j_g"]]
                    .apply(
                        lambda x: applicable(
                            x, target_h=this_target_h, cutoff_min=cutoff_min
                        ),
                    )
                    .reset_index(level=0)
                )
                # rename
                _results.columns = ["sample", "cumulated_heat_at_hours"]
                _results["target_h"] = this_target_h
                _results["cutoff_min"] = cutoff_min
                # append to list
                list_of_results.append(_results)
            # build overall results DataFrame
            results = pd.concat(list_of_results)

        # return
        return results

    #
    # find peaks
    #
    def get_peaks(
        self,
        processparams,
        target_col="normalized_heat_flow_w_g",
        regex=None,
        cutoff_min=None,
        show_plot=True,
        plt_right_s=2e5,
        plt_top=1e-2,
        ax=None,
        xunit="s",
        plot_labels=None,
        xmarker=False,
    ) -> pd.DataFrame:
        """
        get DataFrame of peak characteristics.

        Parameters
        ----------
        target_col : str, optional
            measured quantity within which peaks are searched for. The default is "normalized_heat_flow_w_g"
        regex : str, optional
            regex pattern to include only certain experimental result files
            during initialization. The default is None.
        cutoff_min : int | float, optional
            Time in minutes below which collected data points are discarded for peak picking
        show_plot : bool, optional
            Flag whether or not to plot peak picking for each sample. The default is True.
        plt_right_s : int | float, optional
            Upper limit of x-axis of in seconds. The default is 2e5.
        plt_top : int | float, optional
            Upper limit of y-axis of. The default is 1e-2.
        ax : matplotlib.axes._axes.Axes | None, optional
            The default is None.

        Returns
        -------
        pd.DataFrame holding peak characterisitcs for each sample.

        """

        # list of peaks
        list_of_peaks_dfs = []

        # loop samples
        for sample, data in self._iter_samples(regex=regex):
            # cutoff
            if processparams.cutoff.cutoff_min:
                # discard points at early age
                data = data.query("time_s >= @processparams.cutoff.cutoff_min * 60")

            # reset index
            data = data.reset_index(drop=True)

            # target_columns
            _age_col = "time_s"
            _target_col = target_col

            # find peaks
            peaks, properties = signal.find_peaks(
                data[_target_col],
                prominence=processparams.peakdetection.prominence,
                distance=processparams.peakdetection.distance,
            )

            # plot?
            if show_plot:
                if xunit == "h":
                    df_copy = data.copy()
                    df_copy[_age_col] = df_copy[_age_col] / 3600
                    plt_right_s = plt_right_s / 3600
                    self._plot_peak_positions(
                        df_copy,
                        ax,
                        _age_col,
                        _target_col,
                        peaks,
                        sample,
                        plt_top,
                        plt_right_s,
                        plot_labels,
                        xmarker,
                    )
                else:
                    self._plot_peak_positions(
                        data,
                        ax,
                        _age_col,
                        _target_col,
                        peaks,
                        sample,
                        plt_top,
                        plt_right_s,
                        plot_labels,
                        xmarker,
                    )

            # compile peak characteristics
            peak_characteristics = pd.concat(
                [
                    data.iloc[peaks, :],
                    pd.DataFrame(
                        properties["prominences"], index=peaks, columns=["prominence"]
                    ),
                    pd.DataFrame({"peak_nr": np.arange((len(peaks)))}, index=peaks),
                ],
                axis=1,
            )

            # append
            list_of_peaks_dfs.append(peak_characteristics)

        # compile peak information
        peaks = pd.concat(list_of_peaks_dfs)

        if isinstance(ax, matplotlib.axes._axes.Axes):
            # return peak list and ax
            return peaks, ax
        else:  # return peak list only
            return peaks

    #
    # get peak onsets
    #
    def get_peak_onsets(
        self,
        target_col="normalized_heat_flow_w_g",
        age_col="time_s",
        time_discarded_s=900,
        rolling=1,
        gradient_threshold=0.0005,
        show_plot=False,
        exclude_discarded_time=False,
        regex=None,
        ax: plt.Axes = None,
    ):
        """
        get peak onsets based on a criterion of minimum gradient

        Parameters
        ----------
        target_col : str, optional
            measured quantity within which peak onsets are searched for. The default is "normalized_heat_flow_w_g"
        age_col : str, optional
            Time unit within which peak onsets are searched for. The default is "time_s"
        time_discarded_s : int | float, optional
            Time in seconds below which collected data points are discarded for peak onset picking. The default is 900.
        rolling : int, optional
            Width of "rolling" window within which the values of "target_col" are averaged. A higher value will introduce a stronger smoothing effect. The default is 1, i.e. no smoothing.
        gradient_threshold : float, optional
            Threshold of slope for identification of a peak onset. For a lower value, earlier peak onsets will be identified. The default is 0.0005.
        show_plot : bool, optional
            Flag whether or not to plot peak picking for each sample. The default is False.
        exclude_discarded_time : bool, optional
            Whether or not to discard the experimental values obtained before "time_discarded_s" also in the visualization. The default is False.
        regex : str, optional
            regex pattern to include only certain experimental result files during initialization. The default is None.
        ax : matplotlib.axes._axes.Axes | None, optional
            The default is None.
        Returns
        -------
        pd.DataFrame holding peak onset characterisitcs for each sample.

        """

        # init list of characteristics
        list_of_characteristics = []

        # loop samples
        for sample, data in self._iter_samples(regex=regex):
            if exclude_discarded_time:
                # exclude
                data = data.query(f"{age_col} >= {time_discarded_s}")

            # reset index
            data = data.reset_index(drop=True)

            # calculate get gradient
            data["gradient"] = pd.Series(
                np.gradient(data[target_col].rolling(rolling).mean(), data[age_col])
            )

            # get relevant points
            characteristics = data.copy()
            # discard initial time
            characteristics = characteristics.query(f"{age_col} >= {time_discarded_s}")
            # look at values with certain gradient only
            characteristics = characteristics.query("gradient > @gradient_threshold")
            # consider first entry exclusively
            characteristics = characteristics.head(1)

            # optional plotting
            if show_plot:
                # if specific axis to plot to is specified
                if isinstance(ax, matplotlib.axes._axes.Axes):
                    # plot heat flow curve
                    p = ax.plot(data[age_col], data[target_col])

                    # add vertical lines
                    for _idx, _row in characteristics.iterrows():
                        # vline
                        ax.axvline(_row.at[age_col], color=p[0].get_color(), alpha=0.3)
                        # add "slope line"
                        ax.axline(
                            (_row.at[age_col], _row.at[target_col]),
                            slope=_row.at["gradient"],
                            color=p[0].get_color(),
                            # color="k",
                            # linewidth=0.2
                            alpha=0.25,
                            linestyle="--",
                        )

                    # cosmetics
                    # ax.set_xscale("log")
                    ax.set_title("Onset for " + pathlib.Path(sample).stem)
                    ax.set_xlabel(age_col)
                    ax.set_ylabel(target_col)

                    ax.fill_between(
                        [ax.get_ylim()[0], time_discarded_s],
                        [ax.get_ylim()[0]] * 2,
                        [ax.get_ylim()[1]] * 2,
                        color="black",
                        alpha=0.35,
                    )

                    # set axis limit
                    ax.set_xlim(left=100)

                else:
                    # plot heat flow curve
                    plt.plot(data[age_col], data[target_col])

                    # add vertical lines
                    for _idx, _row in characteristics.iterrows():
                        # vline
                        plt.axvline(_row.at[age_col], color="red", alpha=0.3)

                    # cosmetics
                    # plt.xscale("log")
                    plt.title("Onset for " + pathlib.Path(sample).stem)
                    plt.xlabel(age_col)
                    plt.ylabel(target_col)

                    # get axis
                    ax = plt.gca()

                    plt.fill_between(
                        [ax.get_ylim()[0], time_discarded_s],
                        [ax.get_ylim()[0]] * 2,
                        [ax.get_ylim()[1]] * 2,
                        color="black",
                        alpha=0.35,
                    )

                    # set axis limit
                    plt.xlim(left=100)

            # append to list
            list_of_characteristics.append(characteristics)

        # build overall list
        onset_characteristics = pd.concat(list_of_characteristics)

        # return
        if isinstance(ax, matplotlib.axes._axes.Axes):
            # return onset characteristics and ax
            return onset_characteristics, ax
        else:
            # return onset characteristics exclusively
            return onset_characteristics

    #
    # get maximum slope
    #

    def get_maximum_slope(
        self,
        processparams,
        target_col="normalized_heat_flow_w_g",
        age_col="time_s",
        time_discarded_s=900,
        show_plot=False,
        show_info=True,
        exclude_discarded_time=False,
        regex=None,
        read_start_c3s=False,
        ax=None,
        save_path=None,
        xscale="log",
        xunit="s",
    ):
        """
        The method finds the point in time of the maximum slope. It also calculates the gradient at this point. The method can be controlled by passing a customized ProcessingParameters object for the `processparams` parameter. If no object is passed, the default parameters will be used.

        Parameters
        ----------
        target_col : str, optional
            measured quantity within which peak onsets are searched for. The default is "normalized_heat_flow_w_g"
        age_col : str, optional
            Time unit within which peak onsets are searched for. The default is "time_s"
        time_discarded_s : int | float, optional
            Time in seconds below which collected data points are discarded for peak onset picking. The default is 900.
        show_plot : bool, optional
            Flag whether or not to plot peak picking for each sample. The default is False.
        exclude_discarded_time : bool, optional
            Whether or not to discard the experimental values obtained before "time_discarded_s" also in the visualization. The default is False.
        regex : str, optional
            regex pattern to include only certain experimental result files during initialization. The default is None.
        Returns
        -------
        Pandas Dataframe
            A dataframe that contains the time and the gradient of the maximum slope.
        Examples
        --------
        >>> from CaloCem import tacalorimetry as ta
        >>> from pathlib import Path

        >>> thepath = Path(__file__).parent / "data"
        >>> tam = ta.Measurement(thepath)
        >>> processparams = ta.ProcessingParameters()
        >>> processparams..apply = True
        >>> max_slopes = tam.get_maximum_slope(processparams)
        """

        # init list of characteristics
        list_of_characteristics = []

        # loop samples
        for sample, data in self._iter_samples(regex=regex):
            sample_name = pathlib.Path(sample).stem
            if exclude_discarded_time:
                # exclude
                data = data.query(f"{age_col} >= {time_discarded_s}")

            # manual definition of start time to look for c3s - in case auto peak detection becomes difficult
            if read_start_c3s:
                c3s_start_time_s = self._metadata.query(
                    f"sample_number == '{sample_name}'"
                )["t_c3s_min_s"].values[0]
                c3s_end_time_s = self._metadata.query(
                    f"sample_number == '{sample_name}'"
                )["t_c3s_max_s"].values[0]
                data = data.query(
                    f"{age_col} >= {c3s_start_time_s} & {age_col} <= {c3s_end_time_s}"
                )

            if show_info:
                print(f"Determineing maximum slope of {pathlib.Path(sample).stem}")

            processor = HeatFlowProcessor(processparams)

            data = utils.make_equidistant(data)

            if processparams.rolling_mean.apply:
                data = processor.apply_rolling_mean(data)

            data["gradient"], data["curvature"] = (
                processor.calculate_heatflow_derivatives(data)
            )

            characteristics = processor.get_largest_slope(data, processparams)
            if characteristics.empty:
                continue

            # optional plotting
            if show_plot:
                self._plot_maximum_slope(
                    data,
                    ax,
                    age_col,
                    target_col,
                    sample,
                    characteristics,
                    time_discarded_s,
                    save_path=save_path,
                    xscale=xscale,
                    xunit=xunit,
                )
                # plot heat flow curve
                # plt.plot(data[age_col], data[target_col], label=target_col)
                # plt.plot(
                #     data[age_col],
                #     data["gradient"] * 1e4 + 0.001,
                #     label="gradient * 1e4 + 1mW",
                # )

                # # add vertical lines
                # for _idx, _row in characteristics.iterrows():
                #     # vline
                #     plt.axvline(_row.at[age_col], color="green", alpha=0.3)

                # # cosmetics
                # plt.xscale("log")
                # plt.title(f"Maximum slope plot for {pathlib.Path(sample).stem}")
                # plt.xlabel(age_col)
                # plt.ylabel(target_col)
                # plt.legend()

                # # get axis
                # ax = plt.gca()

                # plt.fill_between(
                #     [ax.get_ylim()[0], time_discarded_s],
                #     [ax.get_ylim()[0]] * 2,
                #     [ax.get_ylim()[1]] * 2,
                #     color="black",
                #     alpha=0.35,
                # )

                # # set axis limit
                # plt.xlim(left=100)
                # plt.ylim(bottom=0, top=0.01)

                # # show
                # plt.show()

            # append to list
            list_of_characteristics.append(characteristics)

        if not list_of_characteristics:
            print("No maximum slope found, check you processing parameters")
        # build overall list
        else:
            max_slope_characteristics = pd.concat(list_of_characteristics)
            # return
            return max_slope_characteristics

    
    def get_average_slope(
    self,
    processparams,
    target_col="normalized_heat_flow_w_g",
    age_col="time_s",
    regex=None,
    show_plot=False,
    ax=None,
    save_path=None,
    xscale="linear",
    xunit="s",
    ):
        """
        Calculate average slope by determining 4 additional slope values between 
        onset time and heat flow maximum, in addition to the maximum slope.
        
        Parameters
        ----------
        processparams : ProcessingParameters
            Processing parameters for analysis
        target_col : str, optional
            Target measurement column, by default "normalized_heat_flow_w_g"
        age_col : str, optional
            Time column name, by default "time_s"
        regex : str, optional
            Regex pattern to filter samples, by default None
        show_plot : bool, optional
            Whether to show plots, by default False
        ax : matplotlib.axes.Axes, optional
            Existing axis to plot on, by default None
        save_path : Path, optional
            Path to save plots, by default None
        xscale : str, optional
            X-axis scale, by default "log"
        xunit : str, optional
            Time unit for display, by default "s"
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing average slope characteristics for each sample
        """
        
        # Get maximum slopes using existing method
        max_slopes = self.get_maximum_slope(
            processparams,
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            show_plot=False,
            ax=ax,
            save_path=save_path,
            xscale=xscale,
            xunit=xunit,
        )
        
        if max_slopes is None or max_slopes.empty:
            print("No maximum slopes found. Cannot calculate average slopes.")
            return pd.DataFrame()
        
        # Get onset times using the peak onset method
        onsets = self.get_peak_onset_via_max_slope(
            processparams,
            show_plot=False,
            regex=regex,
            age_col=age_col,
            target_col=target_col,
            xunit=xunit,
        )
        
        if onsets.empty:
            print("No onset times found. Cannot calculate average slopes.")
            return pd.DataFrame()
        
        list_of_characteristics = []
        
        # Loop through samples
        for sample, data in self._iter_samples(regex=regex):
            sample_short = pathlib.Path(sample).stem
            
            # Get max slope data for this sample
            max_slope_row = max_slopes[max_slopes["sample_short"] == sample_short]
            if max_slope_row.empty:
                continue
                
            # Get onset data for this sample
            onset_row = onsets[onsets["sample"] == sample_short]
            if onset_row.empty:
                continue
            
            # Get time points
            onset_time = onset_row["onset_time_s"].iloc[0]
            max_slope_time = max_slope_row[age_col].iloc[0]
            
            # Find heat flow maximum after onset
            data_after_onset = data[data[age_col] >= onset_time]
            if data_after_onset.empty:
                continue
                
            max_hf_time = data_after_onset.loc[data_after_onset[target_col].idxmax(), age_col]
            
            # Create 4 intermediate time points between onset and heat flow maximum
            if max_hf_time <= onset_time:
                print(f"Warning: Heat flow maximum occurs before onset for {sample_short}")
                continue
                
            # Create 6 time points total (onset, 4 intermediate, max_hf)
            time_points = np.linspace(onset_time + 3600, max_hf_time - 3600, 6)
            
            # Calculate slopes at each interval
            slopes = []
            slope_times = []
            
            for i in range(len(time_points) - 1):
                t1, t2 = time_points[i], time_points[i + 1]
                
                # Get data points in this interval
                interval_data = data[(data[age_col] >= t1) & (data[age_col] <= t2)]
                
                if len(interval_data) < 2:
                    continue
                    
                # Calculate slope using linear regression
                x_vals = interval_data[age_col].values
                y_vals = interval_data[target_col].values
                
                # Simple slope calculation: (y2 - y1) / (x2 - x1)
                slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                slopes.append(slope)
                slope_times.append((t1 + t2) / 2)  # Midpoint time
            
            # Include the maximum slope
            max_slope_value = max_slope_row["gradient"].iloc[0]
            slopes.append(max_slope_value)
            slope_times.append(max_slope_time)
            
            # Calculate average slope
            if slopes:
                avg_slope = np.mean(slopes)
                std_slope = np.std(slopes)
                
                # Create characteristics dictionary
                characteristics = {
                    "sample": sample,
                    "sample_short": sample_short,
                    "onset_time_s": onset_time,
                    "max_hf_time_s": max_hf_time,
                    "max_slope_time_s": max_slope_time,
                    "max_slope_value": max_slope_value,
                    "average_slope": avg_slope,
                    "slope_std": std_slope,
                    "n_slopes": len(slopes),
                    "individual_slopes": slopes,
                    "slope_times": slope_times,
                }
                
                # Optional plotting
                if show_plot:
                    self._plot_average_slope_analysis(
                        data, characteristics, ax, age_col, target_col, 
                        sample_short, save_path, xscale, xunit
                    )
                
                list_of_characteristics.append(characteristics)
        
        if not list_of_characteristics:
            print("No average slope characteristics calculated.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        avg_slope_df = pd.DataFrame(list_of_characteristics)
        
        return avg_slope_df


    @staticmethod
    def _plot_average_slope_analysis(
        data, characteristics, ax, age_col, target_col, 
        sample_short, save_path=None, xscale="linear", xunit="s"
    ):
        """Plot average slope analysis for visualization"""
        
        ax, new_ax = utils.create_base_plot(data, ax, age_col, target_col, sample_short, color="gray")
        
        # Plot the heat flow curve
        #ax.plot(data[age_col], data[target_col], 'b-', alpha=0.7, label='Heat Flow')
        
        # Mark onset time
        ax.axvline(characteristics["onset_time_s"], color='green', 
                linestyle='--', alpha=0.7, label='Onset')
        
        # Mark max heat flow time
        ax.axvline(characteristics["max_hf_time_s"], color='orange', 
                linestyle='--', alpha=0.7, label='Max Heat Flow')
        
        # Mark max slope time
        ax.axvline(characteristics["max_slope_time_s"], color='red', 
                linestyle='--', alpha=0.7, label='Max Slope')
        
        # Plot individual slope lines
        colors = plt.cm.viridis(np.linspace(0, 1, len(characteristics["individual_slopes"])))
        
        for i, (slope, time, color) in enumerate(zip(
            characteristics["individual_slopes"], 
            characteristics["slope_times"], 
            colors
        )):
            # Find corresponding y-value
            y_val = np.interp(time, data[age_col], data[target_col])
            
            # Plot slope line (extend ±10% of time range)
            time_range = characteristics["max_hf_time_s"] - characteristics["onset_time_s"]
            dt = 0.1 * time_range
            
            x_line = [time - dt, time + dt]
            y_line = [y_val - slope * dt, y_val + slope * dt]
            
            ax.plot(x_line, y_line, color=color, alpha=0.6, linewidth=2,
                    label=f'Slope {i+1}: {slope:.2e}')
        
        # Add text annotation for average slope
        ax.text(0.05, 0.95, 
                f'Avg Slope: {characteristics["average_slope"]:.2e}\n'
                f'Std: {characteristics["slope_std"]:.2e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xscale(xscale)
        ax.set_xlabel(f'Time [{xunit}]')
        ax.set_ylabel(target_col.replace('_', ' ').title())
        ax.set_title(f'Average Slope Analysis - {sample_short}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        #ax.set_ylim(0,0.003)
        
        if new_ax:
            if save_path:
                plt.tight_layout()
                plt.show()
                #plt.savefig(save_path / f"average_slope_analysis_{sample_short}.pdf")
                plt.close()
            else:
                plt.tight_layout()
                plt.show() 
   
   
    # get reaction onset via maximum slope
    #
    def get_peak_onset_via_max_slope(
        self,
        processparams,
        show_plot=False,
        ax=None,
        regex=None,
        age_col="time_s",
        target_col="normalized_heat_flow_w_g",
        time_discarded_s=900,
        save_path=None,
        xscale="linear",
        xunit="s",
        intersection="dormant_hf",
    ):
        """
        get reaction onset based on tangent of maximum heat flow and heat flow
        during the dormant period. The characteristic time is inferred from
        the intersection of both characteristic lines

        Parameters
        ----------
        show_plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # get onsets
        max_slopes = self.get_maximum_slope(
            processparams,
            regex=regex,
            show_plot=False,
            ax=ax,
        )
        # % get dormant period HFs
        dorm_hfs = self.get_dormant_period_heatflow(
            processparams,  # cutoff_min=cutoff_min, prominence=prominence
            regex=regex,
            show_plot=False,
            # ax=ax,
        )

        # init list
        list_characteristics = []

        # loop samples
        for i, row in max_slopes.iterrows():
            # calculate y-offset
            t = row["normalized_heat_flow_w_g"] - row["time_s"] * row["gradient"]
            # calculate point of intersection
            # calculate x-intersect of tangent with dormant heat flow
            x_intersect_dormant = (
                    float(
                        dorm_hfs[dorm_hfs["sample_short"] == row["sample_short"]][
                            "normalized_heat_flow_w_g"
                        ]
                    )
                    - t
                ) / row["gradient"]
            # elif intersection == "abscissa":
                # calculate x-intersect of tangent with abscissa (y=0)
            x_intersect = row["time_s"] - (row["normalized_heat_flow_w_g"] / row["gradient"])

            data = self._data.query("sample_short == @row['sample_short']")
            sample = row["sample_short"]

            heat_at_intersect = np.interp(x_intersect, data["time_s"], data["normalized_heat_j_g"])

            # append to list
            list_characteristics.append(
                {
                    "sample": row["sample_short"],
                    "onset_time_s_abscissa": x_intersect,
                    "onset_time_min_abscissa": x_intersect / 60,
                    "heat_at_onset_j_g": heat_at_intersect,
                    "onset_time_s": x_intersect_dormant,
                    "onset_time_min": x_intersect_dormant / 60,
                }
            )


            dorm_hfs_sample = dorm_hfs.query("sample_short == @sample")
            # add prefix dorm to all columns
            dorm_hfs_sample.columns = ["dorm_" + s for s in dorm_hfs_sample.columns]

            characteristics = pd.concat([row, dorm_hfs_sample.squeeze()])
            characteristics.loc["xunit"] = xunit
            characteristics.loc["x_intersect"] = x_intersect
            characteristics.loc["intersection"] = intersection
            # print(characteristics.x_intersect)

            # get maximum time value
            tmax = self._data.query("sample_short == @row['sample_short']")[
                "time_s"
            ].max()
            # get maximum heat flow value
            hmax = self._data.query(
                "time_s > 3000 & sample_short == @row['sample_short']"
            )["normalized_heat_flow_w_g"].max()

            if show_plot:
                self._plot_intersection(
                    data,
                    ax,
                    age_col,
                    target_col,
                    sample,
                    # max_slopes,
                    time_discarded_s,
                    characteristics=characteristics,
                    save_path=save_path,
                    xscale=xscale,
                    # xunit=xunit,
                    hmax=hmax,
                    tmax=tmax,
                )
        
        # build overall dataframe to be returned
        onsets = pd.DataFrame(list_characteristics)

        # merge with dorm_hfs
        onsets = onsets.merge(
            dorm_hfs[
                ["sample_short", "normalized_heat_flow_w_g", "normalized_heat_j_g"]
            ],
            left_on="sample",
            right_on="sample_short",
            how="left",
        )

        # rename
        onsets = onsets.rename(
            columns={
                "normalized_heat_flow_w_g": "normalized_heat_flow_w_g_at_dorm_min",
                "normalized_heat_j_g": "normalized_heat_j_g_at_dorm_min",
            }
        )

        # return
        return onsets
        # if isinstance(ax, matplotlib.axes._axes.Axes):
        #     # return onset characteristics and ax
        #     return onsets, ax
        # else:
        #     # return onset characteristics exclusively
        #     return onsets

    #
    # get dormant period heatflow
    #

    def get_dormant_period_heatflow(
        self,
        processparams,
        regex: str = None,
        cutoff_min: int = 5,
        upper_dormant_thresh_w_g: float = 0.002,
        plot_right_boundary=2e5,
        prominence: float = 1e-3,
        show_plot=False,
    ) -> pd.DataFrame:
        """
        get dormant period heatflow

        Parameters
        ----------
        regex : str, optional
            Regex which can be used to filter the data, i.e., only the patterns which fit the regex will be evaluated. The default is None.
        cutoff_min : int | float, optional
            Time at the start of the experiment which will be cutoff from analysis. This can be useful for ex-situ mixed samples. The default is 5.
        upper_dormant_thresh_w_g : float, optional
            Parameter which controls the upper limit for the plotting option. The default is 0.001.
        show_plot : bool, optional
            If set to true, the data is plotted. The default is False.

        Returns
        -------
        Pandas Dataframe

        """

        # init results list
        list_dfs = []

        # loop samples
        for sample, data in self._iter_samples(regex=regex):
            # get peak as "right border"
            _peaks = self.get_peaks(
                processparams,
                # cutoff_min=cutoff_min,
                regex=pathlib.Path(sample).name,
                # prominence=processparams.gradient_peak_prominence, # prominence,
                show_plot=show_plot,
            )

            # identify "dormant period" as range between initial spike
            # and first reaction peak

            if show_plot:
                # plot
                plt.plot(
                    data["time_s"],
                    data["normalized_heat_flow_w_g"],
                    # linestyle="",
                    # marker="o",
                )

            # discard points at early age
            data = data.query("time_s >= @processparams.cutoff.cutoff_min * 60")
            if not _peaks.empty:
                # discard points after the first peak
                data = data.query('time_s <= @_peaks["time_s"].min()')

            # reset index
            data = data.reset_index(drop=True)

            # pick relevant points at minimum heat flow
            data = data.iloc[data["normalized_heat_flow_w_g"].idxmin(), :].to_frame().T

            if show_plot:
                # guide to the eye lines
                plt.axhline(float(data["normalized_heat_flow_w_g"]), color="red")
                plt.axvline(float(data["time_s"]), color="red")
                # indicate cutoff time
                plt.axvspan(0, cutoff_min * 60, color="black", alpha=0.5)
                # limits
                # plt.xlim(0, _peaks["time_s"].min())
                plt.xlim(0, plot_right_boundary)
                plt.ylim(0, upper_dormant_thresh_w_g)
                # title
                plt.title(pathlib.Path(sample).stem)
                # show
                plt.show()

            # add to list
            list_dfs.append(data)

        # convert to overall datafram
        result = pd.concat(list_dfs).reset_index(drop=True)

        # return
        return result

    #
    # get ASTM C1679 characteristics
    #

    def get_astm_c1679_characteristics(
        self,
        processparams,
        individual: bool = False,
        show_plot=False,
        ax=None,
        regex=None,
        xscale="log",
        xunit="s",
    ) -> pd.DataFrame:
        """
        get characteristics according to ASTM C1679. Compiles a list of data
        points at half-maximum "normalized heat flow", wherein the half maximum
        is either determined for each individual heat flow curve individually
        or as the mean value if the heat flow curves considered.

        Parameters
        ----------
        individual : bool, optional
            DESCRIPTION. The default is False.
        processparams: ProcessingParameters
            Dataclass containing parameters which control the processing of the calorimetry data.

        Returns
        -------
        Pandas Dataframe

        Examples
        --------
        Assuming that the calorimetry data is contained in a subfolder `data`, the time according to ASTM c1679 can be obtained by

        >>> from CaloCem import tacalorimetry as ta
        >>> from pathlib import Path
        >>>
        >>> thepath = Path(__file__).parent / "data"
        >>> tam = ta.Measurement(thepath)
        >>> astm = tam.get_astm_c1679_characteristics()
        """

        # get peaks
        peaks = self.get_peaks(processparams, plt_right_s=4e5, show_plot=False)
        # sort peaks by ascending normalized heat flow
        peaks = peaks.sort_values(by="normalized_heat_flow_w_g", ascending=True)
        # select highest peak --> ASTM C1679
        peaks = peaks.groupby(by="sample").last()

        # get data
        data = self.get_data()

        # init empty list for collecting characteristics
        astm_times = []

        # loop samples
        for sample, sample_data in self._iter_samples(regex=regex):
            # pick sample data
            helper = data[data["sample"] == sample]
            helper_df = helper.copy()

            # check if peak was found
            if peaks[peaks["sample_short"] == sample_data.sample_short.iloc[0]].empty:
                helper = helper.iloc[0:1]
                # manually set time to NaN to indicate that no peak was found
                helper["time_s"] = np.nan

            else:
                # restrict to times before the peak
                helper = helper[helper["time_s"] <= peaks.at[sample, "time_s"]]

                # restrict to relevant heatflows the peak
                if individual == True:
                    helper = helper[
                        helper["normalized_heat_flow_w_g"]
                        <= peaks.at[sample, "normalized_heat_flow_w_g"] * 0.50
                    ]
                else:
                    # use half-maximum average
                    helper = helper[
                        helper["normalized_heat_flow_w_g"]
                        <= peaks["normalized_heat_flow_w_g"].mean() * 0.50
                    ]

                # add to list of of selected points
            astm_times.append(helper.tail(1))

            if helper.tail(1)["time_s"].isna().all():
                continue

            if show_plot:
                # plot
                if xunit == "h":
                    helper_df["time_s"] = helper_df["time_s"] / 3600
                    helper["time_s"] = helper["time_s"] / 3600
                if isinstance(ax, matplotlib.axes._axes.Axes):
                    ax.plot(
                        helper_df["time_s"],
                        helper_df["normalized_heat_flow_w_g"],
                        label=sample,
                    )
                    ax.plot(
                        helper.tail(1)["time_s"],
                        helper.tail(1)["normalized_heat_flow_w_g"],
                        marker="o",
                        color="red",
                    )
                    ax.vlines(
                        x=helper.tail(1)["time_s"],
                        ymin=0,
                        ymax=helper.tail(1)["normalized_heat_flow_w_g"],
                        color="red",
                        linestyle="--",
                    )
                    ax.text(
                        x=helper.tail(1)["time_s"],
                        y=helper.tail(1)["normalized_heat_flow_w_g"] / 2,
                        s=r" $t_{ASTM}$ ="
                        + f"{helper.tail(1)['time_s'].values[0]:.1f}",
                        color="red",
                    )
                else:
                    plt.plot(
                        data["time_s"],
                        data["normalized_heat_flow_w_g"],
                        label=sample,
                    )

        # build overall DataFrame
        astm_times = pd.concat(astm_times)

        # return
        return astm_times

    #
    # get data
    #

    def get_data(self):
        """
        A convenience function which returns the Pandas Dataframe containing the read and processed calorimetry data.
        Returns
        -------
        Pandas DataFrame

        Examples
        --------
        Assuming that the calorimetry data is contained in a subfolder `data`, a conventional Pandas dataframe `df` containing the data from all calorimetry files in `data` can be obtained with the following code.

        >>> from CaloCem import tacalorimetry as ta
        >>> from pathlib import Path
        >>>
        >>> thepath = Path(__file__).parent / "data"
        >>> tam = ta.Measurement(thepath)
        >>> df = tam.get_data()

        """

        return self._data

    #
    # get information
    #

    def get_information(self):
        """
        get information

        Returns
        -------
        pd.DataFrame
            information, i.e. date of measurement, operator, comment ...

        """

        return self._info

    #
    # get added metadata
    #
    def get_metadata(self) -> tuple:
        """


         Returns
         -------
        tuple
             pd.DataFrame of metadata and string of the column used as ID (has to
             be unique).
        """

        # return
        return self._metadata, self._metadata_id

    #
    # get sample names
    #

    def get_sample_names(self):
        """
        get list of sample names

        Returns
        -------
        None.

        """

        # get list
        samples = [pathlib.Path(s).stem for s, _ in self._iter_samples()]

        # return
        return samples

    #
    # set
    #

    def normalize_sample_to_mass(
        self, sample_short: str, mass_g: float, show_info=True
    ):
        """
        normalize "heat_flow" to a certain mass

        Parameters
        ----------
        sample_short : str
            "sample_short" name of sample to be normalized.
        mass_g : float
            mass in gram to which "heat_flow_w" are normalized.

        Returns
        -------
        None.

        """

        # normalize "heat_flow_w" to sample mass
        self._data.loc[
            self._data["sample_short"] == sample_short, "normalized_heat_flow_w_g"
        ] = (
            self._data.loc[self._data["sample_short"] == sample_short, "heat_flow_w"]
            / mass_g
        )

        # normalize "heat_j" to sample mass
        try:
            self._data.loc[
                self._data["sample_short"] == sample_short, "normalized_heat_j_g"
            ] = (
                self._data.loc[self._data["sample_short"] == sample_short, "heat_j"]
                / mass_g
            )
        except Exception:
            pass

        # info
        if show_info:
            print(f"Sample {sample_short} normalized to {mass_g}g sample.")

    #
    # infer "heat_j" values
    #

    def _infer_heat_j_column(self):
        """
        helper function to calculate the "heat_j" columns from "heat_flow_w" and
        "time_s" columns

        Returns
        -------
        None.

        """

        # list of dfs
        list_of_dfs = []

        # loop samples
        for sample, roi in self._iter_samples():
            # check whether a "native" "heat_j"-column is available
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
            roi["_h1_y"] = 0.5 * (
                roi["heat_flow_w"] + roi["heat_flow_w"].shift(1)
            ).shift(-1)
            roi["_h2_x"] = (roi["time_s"] - roi["time_s"].shift(1)).shift(-1)

            # integrate
            roi["heat_j"] = (roi["_h1_y"] * roi["_h2_x"]).cumsum()

            # clean
            del roi["_h1_y"], roi["_h2_x"]

            # append to list
            list_of_dfs.append(roi)

        # set data including "heat_j"
        self._data = pd.concat(list_of_dfs)

    #
    # remove pickle files
    #
    def remove_pickle_files(self):
        """
        remove pickle files if re-reading of source files needed

        Returns
        -------
        None.

        """

        # remove files
        for file in [self._file_data_pickle, self._file_info_pickle]:
            # remove file
            pathlib.Path(file).unlink()

    #
    # add metadata
    #
    def add_metadata_source(self, file: str, sample_id_column: str):
        """
        add an additional source of metadata the object. The source file is of
        type "csv" or "xlsx" and holds information on one sample per row. Columns
        can be named without restrictions.

        To allow for a mapping, the values occurring in self._data["sample_short"]
        should appear in the source file. The column is declared via the keyword
        "sample_id_colum"

        Parameters
        ----------
        file : str
            path to additonal metadata source file.
        sample_id_column : str
            column name in the additional source file matching self._data["sample_short"].

        Returns
        -------
        None.

        """

        if not pathlib.Path(file).suffix.lower() in [".csv", ".xlsx"]:
            # info
            print("Please use metadata files of type csv and xlsx only.")
            # return
            return

        # read file
        try:
            # read as Excel
            self._metadata = pd.read_excel(file)
        except ValueError:
            # read as csv
            self._metadata = pd.read_csv(file)

        # save mapper column
        if sample_id_column in self._metadata.columns:
            # save mapper column
            self._metadata_id = sample_id_column
        else:
            # raise custom Exception
            raise AddMetaDataSourceException(self._metadata.columns.tolist())

    #
    # get metadata group-by options
    #
    def get_metadata_grouping_options(self) -> list:
        """
        get a list of categories to group by in in "self.plot_by_category"

        Returns
        -------
        list
            list of categories avaialble for grouping by.
        """

        # get list based on column names of "_metadata"
        return self._metadata.columns.tolist()

    #
    # average by metadata
    #
    def average_by_metadata(
        self,
        group_by: str,
        meta_id="experiment_nr",
        data_id="sample_short",
        time_average_window_s: int = None,
        time_average_log_bin_count: int = None,
        time_s_max: int = 2 * 24 * 60 * 60,
        get_time_from="left",
        resampling_s: str = "5s",
    ):
        """


        Parameters
        ----------
        group_by : str | list[str]
            DESCRIPTION.
        meta_id : TYPE, optional
            DESCRIPTION. The default is "experiment_nr".
        data_id : TYPE, optional
            DESCRIPTION. The default is "sample_short".
        time_average_window_s : TYPE, optional
            DESCRIPTION. The default is 60. The value is not(!) consindered if
            the keyword time_average_log_bin_count is specified
        get_time_from : TYPE, optional
            DESCRIPTION. The default is "left". further options: # "mid" "right"

        time_average_log_bin_count: number of bins if even spacing in logarithmic scale is applied

        Returns
        -------
        None.

        """

        # get metadata
        meta, meta_id = self.get_metadata()

        # get data
        df = self._data

        # make data equidistant grouped by sample_short
        df = (
            df.groupby(data_id)
            .apply(lambda x: utils.apply_resampling(x, resampling_s))
            .reset_index(drop=True)
        )

        # rename sample in "data" by metadata grouping options
        for value, group in meta.groupby(group_by):
            # if one grouping level is used
            if isinstance(value, str) or isinstance(value, int):
                # modify data --> replace "sample_short" with metadata group name
                _idx_to_replace = df[data_id].isin(group[meta_id])
                df.loc[_idx_to_replace, data_id] = str(value)
            # if multiple grouping levels are used
            elif isinstance(value, tuple):
                # modify data --> replace "sample_short" with metadata group name
                _idx_to_replace = df[data_id].isin(group[meta_id])
                df.loc[_idx_to_replace, data_id] = " | ".join([str(x) for x in value])
            else:
                pass

        # sort experimentally detected times to "bins"
        if time_average_log_bin_count:
            # evenly spaced bins on log scale (geometric spacing)
            df["BIN"] = pd.cut(
                df["time_s"],
                np.geomspace(1, time_s_max, num=time_average_log_bin_count),
            )
        elif time_average_window_s:
            # evenly spaced bins on linear scale with fixed width
            df["BIN"] = pd.cut(
                df["time_s"], np.arange(0, time_s_max, time_average_window_s)
            )

        if "BIN" in df.columns:
            # calculate average and std
            df = (
                df.groupby([data_id, "BIN"])
                .agg(
                    {
                        "normalized_heat_flow_w_g": ["mean", "std"],
                        "normalized_heat_j_g": ["mean", "std"],
                    }
                )
                .dropna(thresh=2)
                .reset_index()
            )
        else:
            # calculate average and std
            df = (
                df.groupby([data_id, "time_s"])
                .agg(
                    {
                        "normalized_heat_flow_w_g": ["mean", "std"],
                        "normalized_heat_j_g": ["mean", "std"],
                    }
                )
                .dropna(thresh=2)
                .reset_index()
            )

        # "flatten" column names
        df.columns = ["_".join(i).replace("mean", "_").strip("_") for i in df.columns]

        if "BIN" in df.columns:
            # regain "time_s" columns
            if get_time_from == "left":
                df["time_s"] = [i.left for i in df["BIN"]]
            elif get_time_from == "mid":
                df["time_s"] = [i.mid for i in df["BIN"]]
            elif get_time_from == "right":
                df["time_s"] = [i.right for i in df["BIN"]]

            # remove "BIN" auxiliary column
            del df["BIN"]

        # copy information to "sample" column --> needed for plotting
        df["sample"] = df[data_id]

        # overwrite data with averaged data
        self._data = df

    #
    # undo action of "average_by_metadata"
    #
    def undo_average_by_metadata(self):
        """
        undo action of "average_by_metadata"
        """

        # set "unprocessed" data as exeperimental data / "de-average"
        if not self._data_unprocessed.empty:
            # reset
            self._data = self._data_unprocessed.copy()

    #
    # apply_tian_correction
    #
    def apply_tian_correction(
        self,
        processparams,  # tau=300, window=11, polynom=3, spline_smoothing_1st: float = 1e-9, spline_smoothing_2nd: float = 1e-9
    ) -> None:
        """
        apply_tian_correction

        Parameters
        ----------

        processparams :
            ProcessingParameters object containing all processing parameters for calorimetry data.
        Returns
        -------
        None.

        """

        # apply the correction for each sample
        for s, d in self._iter_samples():
            # get y-data
            y = d["normalized_heat_flow_w_g"]
            # NaN-handling in y-data
            y = y.fillna(0)
            # get x-data
            x = d["time_s"]

            processor = HeatFlowProcessor(processparams)

            dydx, dy2dx2 = processor.calculate_heatflow_derivatives(d)

            if processparams.time_constants.tau2 == None:
                # calculate corrected heatflow
                norm_hf = (
                    dydx * processparams.time_constants.tau1
                    + self._data.loc[
                        self._data["sample"] == s, "normalized_heat_flow_w_g"
                    ]
                )
            else:
                # calculate corrected heatflow
                norm_hf = (
                    dydx
                    * (
                        processparams.time_constants.tau1
                        + processparams.time_constants.tau2
                    )
                    + dy2dx2
                    * processparams.time_constants.tau1
                    * processparams.time_constants.tau2
                    + d["normalized_heat_flow_w_g"]
                )

            self._data.loc[
                self._data["sample"] == s, "normalized_heat_flow_w_g_tian"
            ] = norm_hf

            self._data.loc[
                self._data["sample"] == s, "gradient_normalized_heat_flow_w_g"
            ] = dydx

            # calculate corresponding cumulative heat
            self._data.loc[self._data["sample"] == s, "normalized_heat_j_g_tian"] = (
                integrate.cumulative_trapezoid(norm_hf.fillna(0), x=x, initial=0)
            )

    #
    # undo Tian-correction
    #
    def undo_tian_correction(self):
        """
        undo_tian_correction; i.e. restore original data


        Returns
        -------
        None.

        """

        # call original restore function
        self.undo_average_by_metadata()

    def _apply_adaptive_downsampling(self):
        """
        apply adaptive downsampling to data
        """

        # define temporary empty DataFrame
        df = pd.DataFrame()

        # apply the correction for each sample
        for s, d in self._iter_samples():
            # print(d.sample_short[0])
            # print(len(d))
            d = d.dropna(subset=["normalized_heat_flow_w_g"])

            processor = HeatFlowProcessor(self.processparams)
            d = processor.restrict_data_range(d)
            # apply adaptive downsampling
            if not self.processparams.downsample.section_split:
                d = utils.adaptive_downsample(
                    d,
                    x_col="time_s",
                    y_col="normalized_heat_flow_w_g",
                    processparams=self.processparams,
                )
            else:
                d = utils.downsample_sections(
                    d,
                    x_col="time_s",
                    y_col="normalized_heat_flow_w_g",
                    processparams=self.processparams,
                )
            df = pd.concat([df, d])

        # set data to downsampled data
        self._data = df

    def get_ascending_flank_tangent(
        self,
        processparams,
        target_col="normalized_heat_flow_w_g",
        age_col="time_s",
        flank_fraction_start=0.2,  # Start at 20% of peak height
        flank_fraction_end=0.8,  # End at 80% of peak height
        window_size=0.1,  # Window size as fraction of flank range
        cutoff_min=None,  # Initial cutoff time in minutes to ignore
        show_plot=False,
        regex=None,
        plotpath=None,
    ):
        """
        Determine tangent to ascending flank of peak by averaging over sections.

        Parameters
        ----------
        target_col : str
            Column containing heat flow data
        age_col : str
            Column containing time data
        flank_fraction_start : float
            Start of flank section as fraction of peak height (0-1)
        flank_fraction_end : float
            End of flank section as fraction of peak height (0-1)
        window_size : float
            Size of averaging window as fraction of flank time range
        cutoff_min : float, optional
            Initial cutoff time in minutes to ignore from analysis. If None,
            uses processparams.cutoff.cutoff_min. The default is None.
        show_plot : bool
            Whether to plot the results
        regex : str
            Regex to filter samples

        Returns
        -------
        pd.DataFrame
            DataFrame with tangent characteristics for each sample
        """

        results = []

        for sample, data in self._iter_samples(regex=regex):
            # Apply cutoff if specified - use parameter cutoff_min first, then fallback to processparams
            cutoff_time_min = (
                cutoff_min
                if cutoff_min is not None
                else processparams.cutoff.cutoff_min
            )
            if cutoff_time_min:
                data = data.query(f"{age_col} >= @cutoff_time_min * 60")

            data = data.reset_index(drop=True)

            # Find the main peak
            peaks, _ = signal.find_peaks(
                data[target_col],
                prominence=processparams.peakdetection.prominence,
                distance=processparams.peakdetection.distance,
            )

            if len(peaks) == 0:
                print(f"No peak found in {sample}")
                continue

            # Use the highest peak
            peak_idx = peaks[np.argmax(data.iloc[peaks][target_col])]
            peak_time = data.iloc[peak_idx][age_col]
            peak_value = data.iloc[peak_idx][target_col]

            # Find baseline (minimum before peak)
            baseline_data = data[data[age_col] < peak_time]
            if len(baseline_data) == 0:
                baseline_value = 0
            else:
                baseline_value = baseline_data[target_col].min()

            # Define flank region
            flank_height_range = peak_value - baseline_value
            flank_start_value = (
                baseline_value + flank_fraction_start * flank_height_range
            )
            flank_end_value = baseline_value + flank_fraction_end * flank_height_range

            # Calculate gradient to ensure we only consider regions with positive slope
            data['gradient'] = np.gradient(data[target_col], data[age_col])
            
            # Extract ascending flank data - only include points with positive gradient
            flank_data = data[
                (data[target_col] >= flank_start_value)
                & (data[target_col] <= flank_end_value)
                & (data[age_col] <= peak_time)
                & (data['gradient'] > 0)  # Only positive gradients
            ].copy()
            
            # If no positive gradient data in initial range, try to find the lowest point with positive gradient
            if len(flank_data) < 3:
                # Find data points with positive gradient before peak
                positive_gradient_data = data[
                    (data[age_col] <= peak_time) & (data['gradient'] > 0)
                ]
                
                if len(positive_gradient_data) >= 3:
                    # Adjust flank start to the minimum value with positive gradient
                    min_positive_value = positive_gradient_data[target_col].min()
                    adjusted_flank_start = max(flank_start_value, min_positive_value)
                    
                    flank_data = data[
                        (data[target_col] >= adjusted_flank_start)
                        & (data[target_col] <= flank_end_value)
                        & (data[age_col] <= peak_time)
                        & (data['gradient'] > 0)
                    ].copy()
                    
                    # Update the flank_start_value for recording
                    flank_start_value = adjusted_flank_start

            if len(flank_data) < 3:
                print(f"Insufficient flank data in {sample}")
                continue

            # Calculate moving tangents over windows
            flank_time_range = flank_data[age_col].max() - flank_data[age_col].min()
            window_time = window_size * flank_time_range

            tangent_slopes = []
            tangent_times = []
            tangent_values = []

            # Slide window across flank
            start_time = flank_data[age_col].min()
            end_time = flank_data[age_col].max() - window_time

            step_size = window_time * 0.1  # 10% overlap
            current_time = start_time

            while current_time <= end_time:
                window_end = current_time + window_time
                window_data = flank_data[
                    (flank_data[age_col] >= current_time)
                    & (flank_data[age_col] <= window_end)
                ]

                if len(window_data) >= 3:
                    # Linear regression for this window
                    x = window_data[age_col].values
                    y = window_data[target_col].values

                    # Use numpy polyfit for linear regression
                    slope, intercept = np.polyfit(x, y, 1)

                    # Only consider positive gradients (ascending flank)
                    if slope > 0:
                        # Store tangent info at window center
                        center_time = (current_time + window_end) / 2
                        center_value = slope * center_time + intercept

                        tangent_slopes.append(slope)
                        tangent_times.append(center_time)
                        tangent_values.append(center_value)

                current_time += step_size

            if not tangent_slopes:
                print(
                    f"No valid tangent windows with positive gradients found in {sample}"
                )
                continue

            # Calculate representative tangent (median to avoid outliers)
            representative_slope = np.median(tangent_slopes)
            representative_time = np.median(tangent_times)
            representative_value = np.median(tangent_values)

            # Calculate tangent line parameters
            # y = mx + b, so b = y - mx
            tangent_intercept = (
                representative_value - representative_slope * representative_time
            )
            # calculate x intection
            # y=0, so x = -b/m
            x_intersection = (
                -tangent_intercept / representative_slope if representative_slope != 0 else np.nan
            )
            
            # Calculate intersection with horizontal line at minimum before tangent_time_s
            data_before_tangent = data[data[age_col] <= representative_time]
            if len(data_before_tangent) > 0:
                min_value_before_tangent = data_before_tangent[target_col].min()
                # Intersection: y = min_value = slope * x + intercept
                # x = (y - intercept) / slope
                x_intersection_min = (
                    (min_value_before_tangent - tangent_intercept) / representative_slope 
                    if representative_slope != 0 else np.nan
                )
            else:
                min_value_before_tangent = np.nan
                x_intersection_min = np.nan

            result = {
                "sample": sample,
                "sample_short": pathlib.Path(sample).stem,
                "peak_time_s": peak_time,
                "peak_value": peak_value,
                "tangent_slope": representative_slope,
                "tangent_time_s": representative_time,
                "tangent_value": representative_value,
                "tangent_intercept": tangent_intercept,
                "flank_start_value": flank_start_value,
                "flank_end_value": flank_end_value,
                "n_windows": len(tangent_slopes),
                "slope_std": np.std(tangent_slopes),
                "x_intersection": x_intersection,
                "min_value_before_tangent": min_value_before_tangent,
                "x_intersection_min": x_intersection_min,
            }

            results.append(result)

            # Optional plotting
            if show_plot:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                # Plot 1: Full curve with peak and flank region
                ax1.plot(data[age_col], data[target_col], "b-", alpha=0.7, label="Data")
                ax1.axvline(
                    peak_time, color="red", linestyle="--", alpha=0.7, label="Peak"
                )
                ax1.axhline(flank_start_value, color="green", linestyle=":", alpha=0.7)
                ax1.axhline(flank_end_value, color="green", linestyle=":", alpha=0.7)
                ax1.fill_between(
                    flank_data[age_col],
                    flank_start_value,
                    flank_end_value,
                    alpha=0.2,
                    color="green",
                    label="Flank region",
                )

                # Plot tangent line
                x_tangent = np.linspace(x_intersection, peak_time, 10)
                y_tangent = representative_slope * x_tangent + tangent_intercept
                ax1.plot(
                    x_tangent, y_tangent, "r-", linewidth=2, label="Average tangent"
                )

                # Add text label for tangent slope
                mid_x = (x_intersection + peak_time) / 2
                mid_y = representative_slope * mid_x + tangent_intercept
                ax1.annotate(
                    f"Slope: {representative_slope:.2e}",
                    xy=(mid_x, mid_y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

                # Add vertical line and label for x_intersection
                if not np.isnan(x_intersection) and x_intersection > data[age_col].min():
                    ax1.axvline(x_intersection, color='orange', linestyle=':', alpha=0.8,) 
                            #    label=f'X-intersection: {x_intersection:.0f}s')
                    ax1.annotate(
                        f"{x_intersection:.0f}s",
                        xy=(x_intersection, baseline_value),
                        xytext=(-50, 20),
                        textcoords='offset points',
                        fontsize=10,
                        color='orange',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.6)
                    )

                # Add horizontal line at minimum and its intersection with tangent
                if not np.isnan(min_value_before_tangent) and not np.isnan(x_intersection_min):
                    # Draw horizontal line at minimum value
                    ax1.axhline(min_value_before_tangent, color='purple', linestyle='--', 
                               alpha=0.7, label=f'Min before tangent: {min_value_before_tangent:.4f}')
                    
                    # Add vertical line at intersection with minimum
                    if x_intersection_min > data[age_col].min() and x_intersection_min < peak_time:
                        ax1.axvline(x_intersection_min, color='purple', linestyle=':', alpha=0.8)
                        ax1.annotate(
                            f"Min-int: {x_intersection_min:.0f}s",
                            xy=(x_intersection_min, min_value_before_tangent),
                            xytext=(10, -30),
                            textcoords='offset points',
                            fontsize=10,
                            color='purple',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color='purple', alpha=0.6)
                        )

                ax1.set_xlabel(age_col)
                ax1.set_ylabel(target_col)
                ax1.set_title(f"Peak Analysis: {pathlib.Path(sample).stem}")
                ax1.legend()
                # ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0,)

                # Plot 2: Slope variation across windows
                ax2.plot(tangent_times, tangent_slopes, "bo-", alpha=0.7)
                ax2.axhline(
                    representative_slope,
                    color="red",
                    linestyle="--",
                    label=f"Median slope: {representative_slope:.2e}",
                )
                ax2.set_xlabel("Window center time (s)")
                ax2.set_ylabel("Local slope")
                ax2.set_title("Slope variation across flank windows")
                ax2.legend()
                # ax2.grid(True, alpha=0.3)
                # ax2.set_ylim(0,)

                plt.tight_layout()
                if plotpath is not None:
                    filename = pathlib.Path(sample).stem
                    plt.savefig(plotpath / f"{filename}.png")
                else:
                    plt.show()

        return pd.DataFrame(results)


class HeatFlowProcessor:
    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def get_largest_slope(
        self, df: pd.DataFrame, processparams: ProcessingParameters
    ) -> pd.DataFrame:
        peak_list = signal.find_peaks(
            df["gradient"],
            distance=processparams.gradient_peakdetection.distance,
            width=processparams.gradient_peakdetection.width,
            rel_height=processparams.gradient_peakdetection.rel_height,
            prominence=processparams.gradient_peakdetection.prominence,
            height=processparams.gradient_peakdetection.height,
        )

        if len(peak_list[0]) == 0:
            print("No peak in gradient found, check your ProcessingParameters")
            return pd.DataFrame()

        if processparams.gradient_peakdetection.use_first:
            # get first found index
            idx = peak_list[0][0]
        else:
            idx = peak_list[0]

        # get index corresponding to maximum gradient

        # consider first entry exclusively
        if processparams.gradient_peakdetection.use_first:
            df = df.iloc[idx, :].to_frame().T

        elif (
            processparams.gradient_peakdetection.use_largest_width
        ):  # use_largest_gradient_peak_width:
            if len(peak_list[1]["widths"]) == 0:
                print("No peak found")
            else:
                gradient_peak_widths = peak_list[1]["widths"]
                idx_max_width = np.argmax(gradient_peak_widths)
                idx_max = peak_list[0][idx_max_width]
                df = df.iloc[idx_max, :].to_frame().T

        elif processparams.gradient_peakdetection.use_largest_width_height:
            if len(peak_list[1]["widths"]) == 0:
                print("No peak found")
            else:
                gradient_peak_width_height = peak_list[1]["width_heights"]
                idx_max_width_height = np.argmax(gradient_peak_width_height)
                idx_max = peak_list[0][idx_max_width_height]
                df = df.iloc[idx_max, :].to_frame().T

        else:
            df = df.iloc[idx, :]

        return df

    def apply_rolling_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        df["td"] = pd.to_timedelta(df["time_s"], unit="s")
        df["normalized_heat_flow_w_g"] = df.rolling(
            self.processparams.rolling_mean.window, on="td"
        )["normalized_heat_flow_w_g"].mean()
        return df

    def apply_median_filter(self, df: pd.DataFrame, deriv_order: str) -> pd.DataFrame:
        df.loc[:, deriv_order] = median_filter(
            df[deriv_order], self.processparams.median_filter.size
        )
        return df

    def apply_spline_interpolation(
        self, df: pd.DataFrame, deriv_order: str
    ) -> pd.DataFrame:
        f = UnivariateSpline(
            df["time_s"],
            df[deriv_order],
            k=3,
            # FIX THIS -- SAME SMOOTHING FOR BOTH DERIVS
            s=self.processparams.spline_interpolation.smoothing_1st_deriv,
            ext=1,
        )
        df.loc[:, deriv_order] = f(df["time_s"])
        return df

    def calculate_hf_derivative(self, df: pd.DataFrame, order: str) -> pd.DataFrame:
        deriv_order = f"{order}_derivative"

        if order == "first":
            df.loc[:, deriv_order] = np.gradient(
                df["normalized_heat_flow_w_g"], df["time_s"]
            )
        elif order == "second":
            df.loc[:, deriv_order] = np.gradient(df["first_derivative"], df["time_s"])

        if self.processparams.median_filter.apply:
            df = self.apply_median_filter(df, deriv_order)

        if self.processparams.spline_interpolation.apply:
            df = df.dropna(subset=[deriv_order])
            df = self.apply_spline_interpolation(df, deriv_order)

        return df

    def calculate_heatflow_derivatives(self, df: pd.DataFrame) -> tuple:
        df = self.calculate_hf_derivative(df, "first").copy()
        df = self.calculate_hf_derivative(df, "second").copy()

        return df["first_derivative"], df["second_derivative"]

    def restrict_data_range(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.query(
            f"time_s >= {self.processparams.cutoff.cutoff_min * 60} & time_s <= {self.processparams.cutoff.cutoff_max * 60}"
        )
