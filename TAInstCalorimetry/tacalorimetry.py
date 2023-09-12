import csv
import logging
import os
import pathlib
import pickle
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysnooper
from scipy import signal
from scipy.interpolate import splrep, splev

from TAInstCalorimetry import utils

logging.basicConfig(
    filename="TAInstCalorimetry.log",
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
    Base class of "tacalorimetry"
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
        self, folder=None, show_info=False, regex=None, auto_clean=True, cold_start=True
    ):
        """
        intialize measurements from folder

        Parameters
        ----------
        folder : str, optional
            path to folder containing .xls and/or .csv experimental result
            files. The default is None.
        show_info : bool, optional
            whether or not to print some informative lines during code
            execution. The default is False.
        regex : str, optional
            regex pattern to include only certain experimental result files
            during initialization. The default is None.
        auto_clean : bool, optional
            whether or not to exclude NaN values contained in the original
            files and combine data from differently names temperature columns.
            The default is True.
        cold_start : bool, optional
            whether or not to use "pickled" files for initialization; save time
            on reading
        Returns
        -------
        None.

        """

        # read
        if folder:
            if cold_start:
                # get data and parameters
                self.get_data_and_parameters_from_folder(
                    folder, regex=regex, show_info=show_info
                )
            else:
                # get data and parameters from pickled files
                self.get_data_and_parameters_from_pickle()
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

        # Message
        print(
            "================\nAre you missing some samples? Try rerunning with auto_clean=True and cold_start=True.\n================="
        )

    #
    # get_data_and_parameters_from_folder
    #
    def get_data_and_parameters_from_folder(self, folder, regex=None, show_info=True):
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
                        self._info = self._read_calo_info_xls(file, show_info=show_info)

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
                        self._data = self._read_calo_data_xls(file, show_info=show_info)

            # append csv
            if f.endswith(".csv"):
                # collect data
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
                        self._data = self._read_calo_data_csv(file, show_info=show_info)

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
        try:
            self._infer_heat_j_column()
        except Exception:
            pass

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
    def get_data_and_parameters_from_pickle(self):
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
        # nr_lines = empty_lines[1] - empty_lines[0] - 2
        return empty_lines

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

        # get "column" count
        data["count"] = [len(i) for i in data[0].str.split(",")]

        # get most frequent count --> assume this for selection of "data" rows
        data = data.loc[data["count"] == data["count"].value_counts().index[0], [0]]

        # init and loop list of lists
        list_of_lists = []
        for _, r in data.iterrows():
            # append to list
            list_of_lists.append(str(r.to_list()).strip("['']").split(","))

        # get DataFrame from list of lists
        data = pd.DataFrame(list_of_lists)

        # get new column names
        new_columnames = []
        for i in data.iloc[0, :]:
            # build
            new_columname = (
                re.sub(r'[\s\n\[\]\(\)° _"]+', "_", i.lower())
                .replace("/", "_")
                .replace("_signal_", "_")
                .strip("_")
            )

            # select appropriate unit
            if new_columname == "time":
                new_columname += "_s"
            elif "temperature" in new_columname:
                new_columname += "_c"
            elif new_columname == "heat_flow":
                new_columname += "_w"
            elif new_columname == "heat":
                new_columname += "_j"
            elif new_columname == "normalized_heat_flow":
                new_columname += "_w_g"
            elif new_columname == "normalized_heat":
                new_columname += "_j_g"
            else:
                new_columname += "_nan"

            # add to list
            new_columnames.append(new_columname)

        # set
        data.columns = new_columnames
        
        # validate new column names
        if not "time_s" in new_columnames:
            # stop here
            return None

        # cut out data part
        data = data.iloc[1:, :].reset_index(drop=True)

        # drop column
        try:
            data = data.drop(columns=["time_markers_nan"])
        except KeyError:
            pass

        # remove columns with too many NaNs
        data = data.dropna(axis=1, thresh=3)
        # # remove rows with NaNs
        data = data.dropna(axis=0)

        # float conversion
        for _c in data.columns:
            # convert
            try:
                data[_c] = data[_c].astype(float)
            except ValueError:
                # go to next column
                continue

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
        data["sample"] = file
        data["sample_short"] = pathlib.Path(file).stem

        # type conversion
        data = utils.convert_df_to_float(data)

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
            # get mass
            mass = float(raw.iloc[3,3].replace(",", "."))
        except IndexError:
            # set mass to None
            mass = None
            # go on
            pass
        
        
        # get "reaction start" time (if available)
        try:
            # get "reaction start" time in seconds
            t0 = float(data[data[2].str.lower() == "reaction start"].head(1)[0])
        except Exception:
            # set t0 to None
            t0 = None
            # go on
            pass

        # remove all-Nan columns
        data = data.dropna(how="all", axis=1)
        
        # restrict to first two columns
        data = data.iloc[:,:2]

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
        # remove "heat_flow_w" column
        del data["heat_flow_mw"]

        # take into account time offset via "reactin start" time
        if t0:
            data["time_s"] -= t0

        # calculate normalized heat flow and heat
        if mass:
            data["normalized_heat_flow_w_g"] = data["heat_flow_w"] / mass

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
                print("\n\n===============================================================")
                print(f"{e} in file \'{pathlib.Path(file).name}\'")
                print("Please, rename the data sheet to \'Raw data\' (device default).")
                print("===============================================================\n\n")

            # log
            logging.info(f"\u2716 reading {file} FAILED.")
            
            # return
            return None


    #
    # iterate samples
    #
    def iter_samples(self, regex=None):
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
                    # go to next
                    continue

            # "return"
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
        ax=None
    ):
        """
        plot  of
            - normalizedheatflow
            - normalizedheat

        in SI-units and "practical" units

        with time units of
            - s
            - min
            - h
            - d
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

        
        # if specific axis to plot to is specified
        if isinstance(ax, matplotlib.axes._axes.Axes):
            # iterate samples
            for s, d in self.iter_samples():
                # define pattern
                if regex:
                    if not re.findall(rf"{regex}", os.path.basename(s)):
                        # go to next
                        continue
                # plot

                # "standard" plot --> individual or averaged
                p_mean = ax.plot(
                    d["time_s"] * x_factor,
                    d[y_column] * y_factor,
                    label=os.path.basename(d["sample"].tolist()[0])
                    .split(".xls")[0]
                    .split(".csv")[0],
                )
                try:
                    # std plot of "fill_between" type
                    ax.fill_between(
                        d["time_s"] * x_factor,
                        (d[y_column] - d[f"{y_column}_std"]) * y_factor,
                        (d[y_column] + d[f"{y_column}_std"]) * y_factor,
                        color=p_mean[0].get_color(),
                        alpha=0.4,
                        label=None,
                    )
                except Exception:
                    # do nothing
                    pass

            # legend
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # limits
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            # add labels
            ax.set_xlabel(f"Age / [{t_unit}]")
            ax.set_ylabel(y_label)

            # return ax
            return ax
        
        # if no specific axis to plot to is specified
        else:
            # iterate samples
            for s, d in self.iter_samples():
                # define pattern
                if regex:
                    if not re.findall(rf"{regex}", os.path.basename(s)):
                        # go to next
                        continue
                # plot
    
                # "standard" plot --> individual or averaged
                p_mean = plt.plot(
                    d["time_s"] * x_factor,
                    d[y_column] * y_factor,
                    label=os.path.basename(d["sample"].tolist()[0])
                    .split(".xls")[0]
                    .split(".csv")[0],
                )
                try:
                    # std plot of "fill_between" type
                    plt.fill_between(
                        d["time_s"] * x_factor,
                        (d[y_column] - d[f"{y_column}_std"]) * y_factor,
                        (d[y_column] + d[f"{y_column}_std"]) * y_factor,
                        color=p_mean[0].get_color(),
                        alpha=0.4,
                        label=None,
                    )
                except Exception:
                    # do nothing
                    pass
    
            # legend
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    
            # limits
            plt.xlim(left=0)
            plt.ylim(bottom=0)
    
            # add labels
            plt.xlabel(f"Age / [{t_unit}]")
            plt.ylabel(y_label)
    
            # return ax
            return plt.gca()

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
        category : str, list[str]
            category (from "self.get_metadata_grouping_options") to group by.
            specify a string or a list of strings here
        t_unit : TYPE, optional
            see "self.plot". The default is "h".
        y : TYPE, optional
            see "self.plot". The default is "normalized_heat_flow_w_g".
        y_unit_milli : TYPE, optional
            see "self.plot". The default is True.

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
                title = f"Grouped by { build_helper_string(categories)} ({build_helper_string(selections)})"
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

    #
    # get the cumulated heat flow a at a certain age
    #

    def get_cumulated_heat_at_hours(self, target_h=4, cutoff_min=None):
        """
        get the cumulated heat flow a at a certain age
        """

        def applicable(df, target_h=4, cutoff_min=None):
            # convert target time to seconds
            target_s = 3600 * target_h
            # get heat at target time
            hf_at_target = float(
                df.query("time_s >= @target_s").head(1)["normalized_heat_j_g"]
            )
            #

            # if cutoff time specified
            if cutoff_min:
                # convert target time to seconds
                target_s = 60 * cutoff_min
                try:
                    hf_at_cutoff = float(
                        df.query("time_s <= @target_s").tail(1)["normalized_heat_j_g"]
                    )
                    # correct heatflow for heatflow at cutoff
                    hf_at_target = hf_at_target - hf_at_cutoff
                except TypeError:
                    name_wt_nan = df["sample_short"].tolist()[0]
                    print(
                        f"Found NaN in Normalized heat of sample {name_wt_nan} searching for cumulated heat at {target_h}h and a cutoff of {cutoff_min}min."
                    )
                    return np.NaN

            # return
            return hf_at_target

        # in case of one specified time
        if isinstance(target_h, int) or isinstance(target_h, float):
            # groupby
            results = (
                self._data.groupby(by="sample")
                .apply(
                    lambda x: applicable(x, target_h=target_h, cutoff_min=cutoff_min)
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
                    self._data.groupby(by="sample")
                    .apply(
                        lambda x: applicable(
                            x, target_h=this_target_h, cutoff_min=cutoff_min
                        )
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
        target_col="normalized_heat_flow_w_g",
        regex=None,
        cutoff_min=None,
        prominence=0.001,
        distance=1,
        show_plot=True,
        plt_right_s=2e5,
        plt_top=1e-2,
        ax=None
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
        prominence : float, optional
            Paek prominence characteristic for the identification of peaks. For a lower promince value, more peaks are found. The default is 0.001.
        distance : int, optional
            count experimental data points between accepted peaks
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
        for sample, data in self.iter_samples(regex=regex):

            # cutoff
            if cutoff_min:
                # discard points at early age
                data = data.query("time_s >= @cutoff_min * 60")

            # reset index
            data = data.reset_index(drop=True)

            # target_columns
            _age_col = "time_s"
            _target_col = target_col

            # find peaks
            peaks, properties = signal.find_peaks(
                data[_target_col], prominence=prominence, distance=distance
            )

            # plot?
            if show_plot:
                # if specific axis to plot to is specified
                if isinstance(ax, matplotlib.axes._axes.Axes):
                    # plot
                    ax.plot(data[_age_col], data[_target_col], label=pathlib.Path(sample).stem)
                    ax.plot(
                        data[_age_col][peaks], data[_target_col][peaks], "x", color="red"
                    )
                    ax.vlines(
                        x=data[_age_col][peaks],
                        ymin=0,
                        ymax=data[_target_col][peaks],
                        color="red",
                    )
                    # add "barrier"
                    ax.axvline(15 * 60, color="green", linestyle=":", linewidth=3)
                    # figure cosmetics
                    ax.legend(loc="best")
                    ax.set_xlim(left=0)
                    ax.set_xlim(right=plt_right_s)
                    ax.set_ylim(bottom=0)
                    ax.set_ylim(top=plt_top)
                    ax.set_xlabel(_age_col)
                    ax.set_ylabel(_target_col)
                    ax.set_title("Peak plot for " + pathlib.Path(sample).stem)
                else:
                    # plot
                    plt.plot(data[_age_col], data[_target_col])
                    plt.plot(
                        data[_age_col][peaks], data[_target_col][peaks], "x", color="red"
                    )
                    plt.vlines(
                        x=data[_age_col][peaks],
                        ymin=0,
                        ymax=data[_target_col][peaks],
                        color="red",
                    )
                    # add "barrier"
                    plt.axvline(15 * 60, color="green", linestyle=":", linewidth=3)
                    # figure cosmetics
                    plt.xlim(left=0)
                    plt.xlim(right=plt_right_s)
                    plt.ylim(bottom=0)
                    plt.ylim(top=plt_top)
                    plt.xlabel(_age_col)
                    plt.ylabel(_target_col)
                    plt.title("Peak plot for " + pathlib.Path(sample).stem)
                    # show
                    plt.show()

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
        else: # return peak list only
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
        ax=None
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
        for sample, data in self.iter_samples(regex=regex):

            if exclude_discarded_time:
                # exclude
                data = data.query(f"{age_col} >= {time_discarded_s}")

            # reset index
            data = data.reset_index(drop=True)

            # calculate get gradient
            data["gradient"] = pd.Series(
                np.gradient(
                    data[target_col].rolling(rolling).mean(),
                    data[age_col]
                    )
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
                        ax.axvline(
                            _row.at[age_col], 
                            color=p[0].get_color(), 
                            alpha=0.3
                            )
                        # add "slope line"
                        ax.axline(
                            (_row.at[age_col], _row.at[target_col]),
                            slope=_row.at["gradient"],
                            color=p[0].get_color(),
                            # color="k",
                            # linewidth=0.2
                            alpha=0.25,
                            linestyle="--"
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
        target_col="normalized_heat_flow_w_g",
        age_col="time_s",
        time_discarded_s=900,
        rolling=1,
        show_plot=False,
        exclude_discarded_time=False,
        regex=None,
    ):
        """
        get maximum slope as a characteristic value
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
         show_plot : bool, optional
            Flag whether or not to plot peak picking for each sample. The default is False.
        exclude_discarded_time : bool, optional
            Whether or not to discard the experimental values obtained before "time_discarded_s" also in the visualization. The default is False.
        regex : str, optional
            regex pattern to include only certain experimental result files during initialization. The default is None.
        Returns
        -------
        pd.DataFrame holding peak onset characterisitcs for each sample.
    
        """
    
        # init list of characteristics
        list_of_characteristics = []
    
        # loop samples
        for sample, data in self.iter_samples(regex=regex):
    
            if exclude_discarded_time:
                # exclude
                data = data.query(f"{age_col} >= {time_discarded_s}")
    
            # reset index
            data = data.reset_index(drop=True)
    
            # calculate get gradient
            data["gradient"] = pd.Series(
                np.gradient(
                    data[target_col].rolling(rolling).mean(),
                    data[age_col]                    
                    )
            )
    
            # get relevant points
            characteristics = data.copy()
            # discard initial time
            characteristics = characteristics.query(f"{age_col} >= {time_discarded_s}")
            # get index corresponding to maximum gradient
            idx_max = characteristics["gradient"].idxmax()
            if np.isnan(idx_max):
                # go to next
                continue
            # consider first entry exclusively
            characteristics = characteristics.loc[idx_max,:].to_frame().T
    
            # optional plotting
            if show_plot:
                # plot heat flow curve
                plt.plot(data[age_col], data[target_col])
    
                # add vertical lines
                for _idx, _row in characteristics.iterrows():
                    # vline
                    plt.axvline(_row.at[age_col], color="red", alpha=0.3)
    
                # cosmetics
                plt.xscale("log")
                plt.title(f"Maximum slope plot for {pathlib.Path(sample).stem}")
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
                plt.ylim(bottom=0)
    
                # show
                plt.show()
    
            # append to list
            list_of_characteristics.append(characteristics)
    
        # build overall list
        max_slope_characteristics = pd.concat(list_of_characteristics)

        # return
        return max_slope_characteristics

    #
    # get data
    #

    def get_data(self):
        """
        get data

        Returns
        -------
        pd.DataFrame
            data, i.e. heat flow, heat, sample, ....

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
        samples = [pathlib.Path(s).stem for s, _ in self.iter_samples()]

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
        for sample, roi in self.iter_samples():
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
        sample_id_colum : str
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
        time_average_window_s: int = 60,
        time_average_log_bin_count: int = None,
        time_s_max: int = 2 * 24 * 60 * 60,
        get_time_from="left",
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

        time_average_log_bin_count: number of bins if even spacing in logarithmic
        scale is applied

        Returns
        -------
        None.

        """

        # get metadata
        meta, meta_id = self.get_metadata()

        # get data
        df = self._data

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
        else:
            # evenly spaced bins on linear scale with fixed width
            df["BIN"] = pd.cut(
                df["time_s"], np.arange(0, time_s_max, time_average_window_s)
            )

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

        # "flatten" column names
        df.columns = ["_".join(i).replace("mean", "_").strip("_") for i in df.columns]

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
    def apply_tian_correction(self, tau=300, smoothing=0):
        """
        apply_tian_correction

        Parameters
        ----------
        tau : TYPE, optional
            time constant to be applied for the correction. The value has to 
            be determined experimentally. The default is 300.
        smoothing : TYPE, optional
            smoothing value for spline. A value of 0 implies no modification
            of the experimental data. The default is 0.

        Returns
        -------
        None.

        """
        
        # apply the correction for each sample
        for s, d in self.iter_samples():
            # get y-data
            y = d["normalized_heat_flow_w_g"]
            # NaN-handling in y-data
            y = y.fillna(0)
            # get x-data
            x = d["time_s"]
            
            # Find the B-spline representation of a 1-D curve.
            y = splrep(x, y, k=3, s=smoothing)
            # Evaluate a B-spline or its derivatives
            dydx = splev(x, y, der=1)
            
            self._data.loc[
                self._data["sample"] == s,
                "normalized_heat_flow_w_g_tian"
                ] = dydx*tau + self._data.loc[
                         self._data["sample"] == s, 
                         "normalized_heat_flow_w_g"
                     ]
                    
    
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
                     