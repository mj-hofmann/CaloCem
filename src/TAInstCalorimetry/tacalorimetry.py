import csv
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


#
# Base class of "ta-calorimetry"
#
class Measurement():
    """
    Base class of "tacalorimetry"
    """

    # ensure consistent data column names -- here as class variable
    colnames = [
        "time",
        "ambient_temp_c",
        "temp_c",
        "heat_flow",
        "heat",
        "normalized_heat_flow",
        "normalized_heat",
    ]

    #
    # init
    #
    def __init__(self, folder=None, show_info=False):
        """
        intialize measurements from folder
        """

        # read
        if folder:
            self.get_data_and_parameters_from_folder(
                folder,
                show_info=show_info
            )
        else:
            self._info = None
            self._data = None

    #
    # get_data_and_parameters_from_folder
    #
    def get_data_and_parameters_from_folder(self, folder, show_info=True):
        """
        get_data_and_parameters_from_folder
        """

        # loop folder
        for f in os.listdir(folder):

            if not f.endswith((".xls", ".csv")):
                # go to next
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
                        [self._info, self._read_calo_info_xls(file, show_info=show_info)])
                except:
                    # initialize
                    self._info = self._read_calo_info_xls(file, show_info=show_info)
                
                # collect data
                try:
                    self._data = pd.concat(
                        [self._data, self._read_calo_data_xls(file, show_info=show_info)])
                except:
                    # initialize
                    self._data = self._read_calo_data_xls(file, show_info=show_info)

            # append csv
            if f.endswith(".csv"):
                # collect information
                try:
                    self._data = pd.concat(
                        [self._data, self._read_calo_data_csv(file, show_info=show_info)])
                except:
                    # initialize
                    self._data = self._read_calo_data_csv(file, show_info=show_info)
                
                # collect data
                try:
                    self._info = pd.concat(
                        [self._info, self._read_calo_info_csv(file, show_info=show_info)])
                except:
                    # initialize
                    self._info = self._read_calo_info_csv(file, show_info=show_info)

        # check for "info"
        if not "info" in locals():
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
        # determine number of lines to skip
        empty_lines = self._determine_data_range_csv(file)
        # read data from csv-file
        data = pd.read_csv(
            file, 
            skiprows=empty_lines[0], 
            nrows=empty_lines[1] - empty_lines[0] - 2
        )
        # remove "columns" with many NaNs
        data = data.dropna(axis=1, thresh=10)
        # set column names
        data.columns = self.colnames
        # add sample name as column
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
            file, 
            nrows=empty_lines[0] - 1, 
            names=["parameter", "value"]
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
                sheet_name="Experiment info",
                header=0,
                names=["parameter", "value"]
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
            # get experiment info (first sheet)
            df_data = xl.parse(xl.sheet_names[-1], header=None)

            # remove "columns" with many NaNs
            df_data = df_data.dropna(axis=1, thresh=10)

            # replace init timestamp
            df_data.iloc[0, 0] = "time"

            # get new column names
            new_columnames = []
            for i, j in zip(df_data.iloc[0, :], df_data.iloc[1, :]):
                # build
                new_columnames.append(
                    f"{i}_{j}".lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("°", "_")
                    .replace("__", "_")
                    .replace("\n", "_")
                )

            # set
            df_data.columns = new_columnames

            # cut out data part
            df_data = df_data.iloc[2:, :].reset_index(drop=True)

            # check if ambient temperature is present
            # if not add empty column (after time)
            if not any("ambient" in s for s in new_columnames):
                # print("hjallo")
                df_data.insert(1, "temperature_ambient_c", "nan")

            # try forced renaming
            try:
                df_data.columns = self.colnames
            except Exception as e:
                if show_info:
                    print(e)

            # convert to float
            for c in df_data:
                df_data[c] = df_data[c].astype(float)

            # add sample information
            df_data["sample"] = file

            # rename
            data = df_data

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
    # plot
    #
    def plot(self, t_unit="h", y="normalized_heat_flow", y_unit_milli=True, regex=None, show_info=True):
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
        if y == "normalized_heat_flow":
            y_column = "normalized_heat_flow"
            y_label = "Normalized Heat Flow / [W/g]"
        elif y == "normalized_heat":
            y_column = "normalized_heat"
            y_label = "Normalized Heat / [J/g]"

        if y_unit_milli:
            y_label = y_label.replace("[", "[m")

        # x-unit
        if t_unit == "s":
            x_factor = 1
        elif t_unit == "min":
            x_factor = 1/60
        elif t_unit == "h":
            x_factor = 1/(60*60)
        elif t_unit == "d":
            x_factor = 1/(60*60*24)

        # y-unit
        if y_unit_milli:
            y_factor = 1000
        else:
            y_factor = 1

        # iterate samples
        for s, d in self.iter_samples():
            # define pattern
            if regex:
                if not re.findall(f"{regex}\.xls", os.path.basename(s)):
                    # go to next
                    continue
            # plot
            plt.plot(
                d["time"]*x_factor,
                d[y_column]*y_factor,
                label=os.path.basename(d.loc[0, "sample"]).split(".xls")[0]
            )

        # legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

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
            target_s = 3600*target_h
            # get heat at target time
            hf_at_target = float(
                df.query("time >= @target_s").
                head(1)["normalized_heat"]
            )

            # if cuoff time specified
            if cutoff_min:
                # convert target time to seconds
                target_s = 60*cutoff_min
                hf_at_cutoff = float(
                    df.query("time <= @target_s").
                    tail(1)["normalized_heat"]
                )
                # correct heatflow for heatflow at cutoff
                hf_at_target = hf_at_target - hf_at_cutoff

            # return
            return hf_at_target

        # groupby
        results = self._data.groupby(by="sample").apply(
            lambda x: apllicable(
                x,
                target_h=target_h,
                cutoff_min=cutoff_min
            )
        ).reset_index(level=0)
        # rename
        results.columns = ["sample", "cumulated_heat_at_hours"]
        results["target_h"] = target_h
        results["cutoff_min"] = cutoff_min

        # return
        return results

    #
    # get data
    #

    def get_data(self):
        """
        get data
        """

        return self._data

    #
    # get information
    #

    def get_information(self):
        """
        get information
        """

        return self._info
