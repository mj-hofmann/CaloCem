import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


#
# Base class of "ta-calorimetry"
#
class Measurement:
    """
    Base class of "tacalorimetry"
    """

    #
    # init
    #
    def __init__(self, folder=None, show_info=False, regex=None, auto_clean=True):
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

        Returns
        -------
        None.

        """

        # read
        if folder:
            # get data and parameters
            self.get_data_and_parameters_from_folder(
                folder, regex=regex, show_info=show_info
            )
            if auto_clean:
                # remove NaN values and merge time columns
                self._auto_clean_data()
        else:
            self._info = None
            self._data = None

    #
    # get_data_and_parameters_from_folder
    #
    def get_data_and_parameters_from_folder(self, folder, regex=None, show_info=True):
        """
        get_data_and_parameters_from_folder
        """

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
                    self._data = self._read_calo_data_xls(file, show_info=show_info)

            # append csv
            if f.endswith(".csv"):
                # collect information
                try:
                    self._data = pd.concat(
                        [
                            self._data,
                            self._read_calo_data_csv(file, show_info=show_info),
                        ]
                    )
                except Exception:
                    # initialize
                    self._data = self._read_calo_data_csv(file, show_info=show_info)

                # collect data
                try:
                    self._info = pd.concat(
                        [
                            self._info,
                            self._read_calo_info_csv(file, show_info=show_info),
                        ]
                    )
                except Exception:
                    # initialize
                    self._info = self._read_calo_info_csv(file, show_info=show_info)

        # check for "info"
        if "info" not in locals():
            # set info variable to None
            info = None

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
            data[_c] = data[_c].astype(float)

        # restrict to "time_s" > 0
        data = data.query("time_s >= 0").reset_index(drop=True)

        # add sample information
        data["sample"] = file

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
        # determine number of lines to skip
        empty_lines = self._determine_data_range_csv(file)
        # read info block from csv-file
        info = pd.read_csv(
            file, nrows=empty_lines[0] - 1, names=["parameter", "value"]
        ).dropna(subset=["parameter"])
        # add sample name as column
        info["sample"] = file
        # the last block is not really meta data but summary data and
        # somewhat not necessary

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

            # rename
            data = df_data

            # return
            return data

        except Exception as e:
            if show_info:
                print(e)

    #
    # iterate samples
    #
    def iter_samples(self):
        """
        iterate samples and return corresponding data

        Returns
        -------
        sample (str) : name of the current sample
        data (pd.DataFrame) : data corresponding to the current sample
        """

        for sample, data in self._data.groupby(by="sample"):
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
        elif y == "normalized_heat_j_g":
            y_column = "normalized_heat_j_g"
            y_label = "Normalized Heat / [J/g]"

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

        # iterate samples
        for s, d in self.iter_samples():
            # define pattern
            if regex:
                if not re.findall(rf"{regex}", os.path.basename(s)):
                    # go to next
                    continue
            # plot
            plt.plot(
                d["time_s"] * x_factor,
                d[y_column] * y_factor,
                label=os.path.basename(d["sample"].tolist()[0])
                .split(".xls")[0]
                .split(".csv")[0],
            )

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
    # get the cumulated heat flow a at a certain age
    #
    def get_cumulated_heat_at_hours(self, target_h=4, cutoff_min=None):
        """
        get the cumulated heat flow a at a certain age
        """

        def apllicable(df, target_h=4, cutoff_min=None):
            # convert target time to seconds
            target_s = 3600 * target_h
            # get heat at target time
            hf_at_target = float(
                df.query("time_s >= @target_s").head(1)["normalized_heat_j_g"]
            )

            # if cuoff time specified
            if cutoff_min:
                # convert target time to seconds
                target_s = 60 * cutoff_min
                hf_at_cutoff = float(
                    df.query("time_s <= @target_s").tail(1)["normalized_heat_j_g"]
                )
                # correct heatflow for heatflow at cutoff
                hf_at_target = hf_at_target - hf_at_cutoff

            # return
            return hf_at_target

        # groupby
        results = (
            self._data.groupby(by="sample")
            .apply(lambda x: apllicable(x, target_h=target_h, cutoff_min=cutoff_min))
            .reset_index(level=0)
        )
        # rename
        results.columns = ["sample", "cumulated_heat_at_hours"]
        results["target_h"] = target_h
        results["cutoff_min"] = cutoff_min

        # return
        return results

    #
    # find peaks
    #
    def get_peaks(
        self, target_col="normalized_heat_flow_w_g", prominence=0.001, show_plot=True
    ):
        """
        get DataFrame of peak characteristics

        Parameters
        ----------
        prominence : TYPE, optional
            DESCRIPTION. The default is 0.001.
        show_plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        # list of peaks
        list_of_peaks_dfs = []

        # loop samples
        for sample, data in self.iter_samples():

            # reset index
            data = data.reset_index(drop=True)

            # target_columns
            _age_col = "time_s"
            _target_col = target_col

            # find peaks
            peaks, properties = signal.find_peaks(
                data[_target_col],
                prominence=prominence,
            )

            # plot?
            if show_plot:
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
                plt.xlim(right=20000)
                plt.ylim(bottom=0)
                plt.title(sample)
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

        # return peak list
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
    ):
        """
        get peak onsets based on a criterion of minimum gradient

        Returns
        -------
        None.

        """

        # init list of characteristics
        list_of_characteristics = []

        # loop samples
        for sample, data in self.iter_samples():

            if exclude_discarded_time:
                # exclude
                data = data.query(f"{age_col} >= {time_discarded_s}")

            # reset index
            data = data.reset_index(drop=True)

            # calculate get gradient
            data["gradient"] = pd.Series(
                np.gradient(data[target_col].rolling(rolling).mean())
            )

            # get relevant points
            characteristics = data.copy()
            # discard initial time
            characteristics = characteristics.query(f"{age_col} >= {time_discarded_s}")
            # look at values with certain gradient only
            characteristics = characteristics.query("gradient > @gradient_threshold")

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
                plt.title(sample)

                # get axis
                ax = plt.gca()

                plt.fill_between(
                    [ax.get_ylim()[0], time_discarded_s],
                    [ax.get_ylim()[0]] * 2,
                    [ax.get_ylim()[1]] * 2,
                    color="black",
                    alpha=0.35,
                )

                # show
                plt.show()

            # append to list
            list_of_characteristics.append(characteristics)

        # build overall list
        onset_characteristics = pd.concat(list_of_characteristics)

        # return
        return onset_characteristics

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
