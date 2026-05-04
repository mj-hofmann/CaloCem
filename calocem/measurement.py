"""
Refactored main measurement class for calorimetry data handling.
"""

import logging
import pathlib
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analysis import (
    ASTMC1679Analyzer,
    AverageSlopeAnalyzer,
    DeconvolutionAnalyzer,
    DormantPeriodAnalyzer,
    FirstAscendingSlopeAnalyzer,
    FlankTangentAnalyzer,
    HeatCalculator,
    OnsetAnalyzer,
    PeakAnalyzer,
    SlopeAnalyzer,
)
from .data_processing import DataCleaner, DataNormalizer, HeatFlowProcessor, MetadataAggregator, SampleIterator
from .exceptions import AutoCleanException, ColdStartException, DataProcessingException
from .file_io import DataPersistence, FolderDataLoader
from .plotting import SimplePlotter
from .processparams import ProcessingParameters

logger = logging.getLogger(__name__)


class Measurement:
    """
    Class for handling and processing isothermal heat flow calorimetry data.

    This class coordinates file I/O, data processing, analysis, and visualization
    operations while maintaining the same API as the original implementation.
    """

    def __init__(
        self,
        folder: Optional[Union[str, pathlib.Path]] = None,
        show_info: bool = True,
        regex: Optional[str] = None,
        auto_clean: bool = False,
        cold_start: bool = True,
        processparams: Optional[ProcessingParameters] = None,
        new_code: bool = False,
        processed: bool = False,
        metadata_path: Optional[Union[str, pathlib.Path]] = None,
        metadata_id_column: Optional[str] = None,
        save_cache: bool = False,
    ):
        """
        Initialize measurements from folder or existing data.

        Parameters
        ----------
        folder : str or pathlib.Path, optional
            Path to folder containing experimental files
        show_info : bool, optional
            Whether to print informative messages, by default True
        regex : str, optional
            Regex pattern to filter files, by default None
        auto_clean : bool, optional
            Whether to clean data automatically, by default False
        cold_start : bool, optional
            Whether to read from files or use cached data, by default True
        processparams : ProcessingParameters, optional
            Processing parameters, by default None. If None, the default parameters will be used
        new_code : bool, optional
            Flag for new code features, by default False
        processed : bool, optional
            Whether data is already processed, i.e., if a .csv file is used which was processed  by Calocem. By default False
        metadata_path : str or pathlib.Path, optional
            Path to metadata file (CSV, Excel, etc.), by default None
        metadata_id_column : str, optional
            Column name in metadata file that matches sample names, by default None
        save_cache : bool, optional
            Whether to write `_data.pickle` and `_info.pickle` cache files when loading
            from a folder, by default False. When True, subsequent runs can be sped up
            with ``cold_start=False`` to read from the cache instead of re-parsing the
            folder. When False (default), no pickle files are created.
        """
        # Initialize attributes
        self._data = pd.DataFrame()
        self._info = pd.DataFrame()
        self._data_unprocessed = pd.DataFrame()
        self._metadata = pd.DataFrame()
        self._metadata_id = ""

        # Store configuration
        self._new_code = new_code
        self._processed = processed
        self._save_cache = save_cache

        # Setup processing parameters
        if not isinstance(processparams, ProcessingParameters):
            self.processparams = ProcessingParameters()
        else:
            self.processparams = processparams

        # Initialize components
        self._folder_loader = FolderDataLoader(processed=processed)
        self._data_persistence = DataPersistence()
        self._data_cleaner = DataCleaner()
        self._plotter = SimplePlotter()

        # Load data if folder provided
        if folder:
            try:
                if cold_start:
                    self._load_from_folder(folder, regex, show_info)
                else:
                    self._load_from_cache()

                if auto_clean:
                    self._auto_clean_data()

            except Exception as e:
                if show_info:
                    print(f"Error during initialization: {e}")
                if auto_clean:
                    raise AutoCleanException()
                if not cold_start:
                    raise ColdStartException()
                raise

        # Load metadata if provided
        if metadata_path and metadata_id_column:
            self.add_metadata_source(metadata_path, metadata_id_column, show_info)

        # Apply downsampling if requested
        if self.processparams.downsample.apply:
            self._apply_adaptive_downsampling()

        # Information message
        if show_info:
            print("================")
            print(
                "Are you missing some samples? Try rerunning with auto_clean=True and cold_start=True."
            )
            print("================")

    def _load_from_folder(
        self, folder: Union[str, pathlib.Path], regex: Optional[str], show_info: bool
    ):
        """Load data from folder using file loader."""
        try:
            self._data, self._info = self._folder_loader.load_from_folder(
                folder, regex, show_info
            )
            self._data_unprocessed = self._data.copy()

            # Save to cache only if requested
            if self._save_cache:
                self._data_persistence.save_data(self._data, self._info)

        except Exception as e:
            raise DataProcessingException("load_from_folder", e)

    def _load_from_cache(self):
        """Load data from cached pickle files."""
        try:
            if not self._data_persistence.pickle_files_exist():
                raise FileNotFoundError("No pickle files found for cold start")

            self._data, self._info = self._data_persistence.load_data()
            self._data_unprocessed = self._data.copy()

        except Exception as e:
            raise ColdStartException() from e

    def _auto_clean_data(self):
        """Apply automatic data cleaning."""
        try:
            self._data = self._data_cleaner.auto_clean_data(self._data)
        except Exception as e:
            raise AutoCleanException() from e

    def _apply_adaptive_downsampling(self):
        """Apply adaptive downsampling if configured."""
        # TODO: Implement downsampling logic
        logger.info("Downsampling requested but not yet implemented")

    # Data access methods
    def get_data(self) -> pd.DataFrame:
        """Get the processed calorimetry data."""
        return self._data

    def get_information(self) -> pd.DataFrame:
        """Get the measurement information/metadata."""
        return self._info

    def get_metadata(self) -> tuple:
        """Get added metadata and the ID column name."""
        return self._metadata, self._metadata_id

    def get_sample_names(self) -> list:
        """Get list of sample names."""
        return [
            pathlib.Path(str(sample)).stem
            for sample, _ in SampleIterator.iter_samples(self._data)
        ]

    # Plotting methods
    def plot(
        self,
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
        regex: Optional[str] = None,
        show_info: bool = True,
        ax=None,
    ):
        """Plot the calorimetry data."""
        return self._plotter.plot_data(
            self._data, t_unit, y, y_unit_milli, regex, show_info, ax
        )

    def plot_by_category(
        self,
        categories: str,
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
    ):
        """Plot data by metadata categories."""
        # Simplified implementation - would need full metadata integration
        logger.warning(
            "plot_by_category requires metadata integration - not fully implemented"
        )
        yield from []

    # Analysis methods
    def get_peaks(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        regex: Optional[str] = None,
        cutoff_min: Optional[float] = None,  # Deprecated parameter
        show_plot: bool = True,
        plt_right_s: float = 2e5,
        plt_top: float = 1e-2,
        ax=None,
        xunit: str = "s",
        plot_labels: Optional[bool] = None,
        xmarker: bool = False,
    ) -> pd.DataFrame:
        """Get DataFrame of peak characteristics."""
        if cutoff_min is not None:
            warnings.warn(
                "The cutoff_min parameter is deprecated. Use ProcessingParameters instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        params = processparams or self.processparams
        analyzer = PeakAnalyzer(params)
        peaks_df = analyzer.get_peaks(self._data, target_col, regex)

        if show_plot and not peaks_df.empty:
            # Simple plotting implementation
            for sample, sample_data in SampleIterator.iter_samples(self._data, regex):
                sample_peaks = peaks_df[
                    peaks_df["sample_short"] == pathlib.Path(str(sample)).stem
                ]
                if not sample_peaks.empty:
                    # Locate peak positions within sample_data by matching time_s values
                    peak_indices = np.array([
                        sample_data.index.get_loc(
                            sample_data["time_s"].sub(t).abs().idxmin()
                        )
                        for t in sample_peaks["time_s"]
                    ])
                    self._plotter.plot_peaks(
                        sample_data.reset_index(drop=True),
                        peak_indices,
                        str(sample),
                        ax,
                        "time_s",
                        target_col,
                    )

        return peaks_df

    def get_peak_onsets(
        self,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        time_discarded_s: float = 900,
        rolling: int = 1,
        gradient_threshold: float = 0.0005,
        show_plot: bool = False,
        exclude_discarded_time: bool = False,
        regex: Optional[str] = None,
        ax=None,
    ):
        """Get peak onsets based on gradient threshold."""
        analyzer = OnsetAnalyzer(self.processparams)
        return analyzer.get_peak_onsets(
            self._data,
            target_col,
            age_col,
            time_discarded_s,
            rolling,
            gradient_threshold,
            exclude_discarded_time,
            regex,
        )

    def get_deconvolution(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
        n_peaks: Optional[int] = None,
        peak_shape: str = "lognormal",
        baseline_mode: Optional[str] = None,
        relative_intensity_upper_bounds: Optional[list[float]] = None,
        peak_width_upper_bounds: Optional[list[float]] = None,
        show_plot: bool = False,
        ax=None,
    ) -> pd.DataFrame:
        """Fit a multi-peak deconvolution model to each sample."""
        params = processparams or self.processparams
        analyzer = DeconvolutionAnalyzer(params)
        result = analyzer.get_deconvolution(
            self._data,
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            n_peaks=n_peaks,
            peak_shape=peak_shape,
            baseline_mode=baseline_mode,
            relative_intensity_upper_bounds=relative_intensity_upper_bounds,
            peak_width_upper_bounds=peak_width_upper_bounds,
        )

        if show_plot and not result.empty:
            for sample, sample_data in SampleIterator.iter_samples(self._data, regex):
                sample_short = pathlib.Path(str(sample)).stem
                sample_result = result[result["sample_short"] == sample_short]
                if sample_result.empty:
                    continue

                plot_data = sample_data[[age_col, target_col]].copy()
                if params.cutoff.cutoff_min:
                    plot_data = plot_data[
                        plot_data[age_col] >= params.cutoff.cutoff_min * 60
                    ]
                plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
                if plot_data.empty:
                    continue

                x = plot_data[age_col].to_numpy(dtype=float)
                y = plot_data[target_col].to_numpy(dtype=float)
                x_logn = np.clip(x, 1e-12, None)
                x_range = max(float(np.max(x) - np.min(x)), 1e-12)
                x_scaled = 2.0 * (x - float(np.min(x))) / x_range - 1.0

                created_ax = ax is None
                if created_ax:
                    _, local_ax = plt.subplots(figsize=(7, 5))
                else:
                    local_ax = ax

                local_ax.plot(x, y, color="black", linewidth=1.2, label="data")

                total_components = np.zeros_like(y)
                shape = str(sample_result.iloc[0]["peak_shape"]).lower()
                baseline = str(sample_result.iloc[0]["baseline_mode"]).lower()

                for _, comp in sample_result.iterrows():
                    amplitude = float(comp["amplitude"])
                    center = float(comp["center_time_s"])
                    width = float(comp["width"])

                    if shape == "gaussian":
                        curve = analyzer._gaussian_peak(x, amplitude, center, width)
                    else:
                        curve = analyzer._lognormal_peak(x_logn, amplitude, center, width)

                    total_components += curve
                    local_ax.plot(
                        x,
                        curve,
                        linestyle="--",
                        linewidth=1,
                        label=f"component {int(comp['component'])}",
                    )

                baseline_constant = float(sample_result.iloc[0]["baseline_constant"])
                baseline_slope = float(sample_result.iloc[0]["baseline_slope"])
                if baseline == "constant" and not np.isnan(baseline_constant):
                    total_fit = total_components + baseline_constant
                elif baseline == "linear" and not np.isnan(baseline_constant):
                    slope = 0.0 if np.isnan(baseline_slope) else baseline_slope
                    total_fit = total_components + baseline_constant + slope * x
                elif baseline == "chebyshev":
                    cheb_coeffs = sample_result.iloc[0].get("baseline_cheb_coeffs", None)
                    if isinstance(cheb_coeffs, str):
                        import ast

                        cheb_coeffs = ast.literal_eval(cheb_coeffs)
                    if cheb_coeffs is not None:
                        baseline_curve = np.polynomial.chebyshev.chebval(
                            x_scaled, np.array(cheb_coeffs, dtype=float)
                        )
                        total_fit = total_components + baseline_curve
                    else:
                        total_fit = total_components
                else:
                    total_fit = total_components

                fit_r2 = sample_result["fit_r2"].iloc[0]
                local_ax.plot(
                    x,
                    total_fit,
                    color="tab:red",
                    linewidth=1.5,
                    label=f"fit (R²={fit_r2:.3f})" if not pd.isna(fit_r2) else "fit",
                )
                local_ax.set_title(f"Deconvolution: {sample_short}")
                local_ax.set_xlabel(age_col)
                local_ax.set_ylabel(target_col)
                local_ax.legend()

                if created_ax:
                    plt.show()

        return result

    def get_left_peak_inflection_tangent_intersection(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
        n_peaks: Optional[int] = None,
        peak_shape: str = "lognormal",
        baseline_mode: Optional[str] = None,
        deconvolution_results: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Determine abscissa intersection from tangent at left-peak flank inflection."""
        params = processparams or self.processparams
        analyzer = DeconvolutionAnalyzer(params)

        fit_results = deconvolution_results
        if fit_results is None:
            fit_results = self.get_deconvolution(
                processparams=params,
                target_col=target_col,
                age_col=age_col,
                regex=regex,
                n_peaks=n_peaks,
                peak_shape=peak_shape,
                baseline_mode=baseline_mode,
                show_plot=False,
            )

        return analyzer.get_left_peak_inflection_tangent_intersection(
            self._data,
            fit_results,
            target_col=target_col,
            age_col=age_col,
            regex=regex,
        )

    def get_maximum_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        show_plot: bool = False,
        regex: Optional[str] = None,
        ax=None,
        save_path: Optional[pathlib.Path] = None,
        xunit: str = "s",
    ) -> pd.DataFrame:
        """Find the point in time of the maximum slope.

        Wrapper around :meth:`get_mainpeak_params` returning only the
        ``max_slope`` columns together with sample identifiers.
        """
        result = self.get_mainpeak_params(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            show_plot=show_plot,
            plot_type="max",
            regex=regex,
            ax=ax,
        )
        if result.empty:
            return result
        id_cols = ["sample", "sample_short"]
        max_slope_cols = [c for c in result.columns if "max_slope" in c]
        return result[id_cols + max_slope_cols].reset_index(drop=True)

    def get_mainpeak_params(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        show_plot: bool = False,
        save_plot: bool = False,
        plot_type: str = "mean",
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        ax=None,
        method: str = "mean",
    ) -> pd.DataFrame:
        """
                Unified method for main-peak slope analysis.

                Depending on ``method`` this either:
                - ``"mean"`` (default): calculates maximum and mean slope onset analyses
                - ``"ascending"``: calculates first ascending slope analysis via
                    ``get_first_ascending_slope_to_fraction``

                Results are returned in a single DataFrame with all available slope values.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters, by default None
        target_col : str
            Column containing heat flow data. The default is 'normalized_heat_flow_w_g'.
        age_col : str
            Column containing time data. The default is 'time_s'.
        show_plot : bool
            Whether to plot the results
        plot_type : str
            Type of plot to show: 'max', 'mean',
            - 'max': Shows only maximum slope analysis plot
            - 'mean': Shows only mean slope (flank tangent) analysis plot
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        method : str
            Slope analysis method to run: 'mean' or 'ascending'.

        Returns
        -------
        pd.DataFrame
            Comprehensive DataFrame with available slope and characteristic results.
        
        Examples
        --------
        >>> measurement = Measurement(folder="data/")
        >>> mainpeak_params = measurement.get_mainpeak_params(
        ...     processparams=ProcessingParameters(),
        ...     show_plot=False,
        ...     plot_type="mean",
        ...     method="mean",
        ... )
        """
        params = processparams or self.processparams

        valid_methods = ["mean", "ascending"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        if method == "mean":
            max_slope_results = self._calculate_max_slope_analysis(
                params,
                target_col,
                age_col,
                regex,
            )

            mean_slope_results = self._calculate_mean_slope_analysis(
                params,
                target_col,
                age_col,
                regex,
            )
        elif method == "ascending":
            max_slope_results = pd.DataFrame()
            mean_slope_results = self._calculate_first_ascending_slope_analysis(
                params,
                target_col,
                age_col,
                regex,
            )

        dormant_minimum_heatflow = self.get_dormant_period_heatflow(
            params, regex, show_plot=False
        )

        astm_values = self.get_astm_c1679_characteristics(params, individual=True, show_plot=False, regex=regex)

        # Merge results into comprehensive DataFrame
        combined_results = self._merge_slope_results(
            max_slope_results, mean_slope_results, dormant_minimum_heatflow, astm_values
        )

        # Plot if requested
        if (
            method == "mean"
            and (show_plot or save_plot)
            and not (mean_slope_results.empty or max_slope_results.empty)
        ):
            self._plot_combined_slope_analysis(
                combined_results,
                params,
                target_col,
                age_col,
                plot_type,
                regex,
                plotpath if save_plot else None,
                ax,
                show_plot=show_plot,
            )
            # if not ax:
            #     plt.show()
                # if (save_plot and show_plot) and plotpath:
                #     plt.savefig(plotpath)
                #     plt.show()
                # elif show_plot:
                #     plt.show()
                # elif save_plot:
                #     plt.savefig(plotpath)

        elif (
            method == "ascending"
            and (show_plot or save_plot)
            and not mean_slope_results.empty
        ):
            self._plot_ascending_slope_analysis(
                combined_results,
                params,
                target_col,
                age_col,
                regex,
                plotpath if save_plot else None,
                ax,
                show_plot=show_plot,
            )

        elif method == "mean" and mean_slope_results.empty:
            # logger.warning("No slope analysis results to plot.")
            print("No mean slope analysis obtained - check the processing parameters.")

        elif method == "mean" and max_slope_results.empty:
            print(
                "No maximum slope analysis obtained - check the processing parameters."
            )

        return combined_results

    def _calculate_first_ascending_slope_analysis(
        self,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
    ) -> pd.DataFrame:
        """Calculate first ascending slope analysis and return structured results."""
        analyzer = FirstAscendingSlopeAnalyzer(params)
        first_ascending_results = analyzer.get_first_ascending_slope_to_fraction(
            self._data,
            target_col=target_col,
            age_col=age_col,
            fraction_of_max=params.slope_analysis.first_ascending_fraction_of_max,
            regex=regex,
        )

        if first_ascending_results.empty:
            logger.warning("No first ascending slope results found.")
            return pd.DataFrame()

        results = []
        for _, row in first_ascending_results.iterrows():
            sample = row.get("sample", row.get("sample_short", ""))
            sample_short = row.get("sample_short", row.get("sample", ""))

            representative_slope = row.get("first_ascending_slope")
            tangent_intercept = row.get("first_ascending_intercept")
            tangent_time_s = row.get("first_ascending_tangent_time_s")

            x_intersection = np.nan
            x_intersection_dormant = np.nan
            x_intersection_j_g = np.nan
            x_intersection_dormant_j_g = np.nan
            min_value_before_tangent = np.nan

            sample_data = self._get_filtered_sample_data(
                sample_short,
                age_col,
                cutoff_time_min=params.cutoff.cutoff_min,
            )

            peak_time_s = np.nan
            peak_value = np.nan
            peak_j_g = np.nan
            if not sample_data.empty and target_col in sample_data.columns:
                peak_idx = sample_data[target_col].idxmax()
                if pd.notna(peak_idx):
                    peak_row = sample_data.loc[peak_idx]
                    peak_time_s = float(peak_row[age_col])
                    peak_value = float(peak_row[target_col])
                    if "normalized_heat_j_g" in sample_data.columns:
                        peak_j_g = float(peak_row["normalized_heat_j_g"])

            if (
                not sample_data.empty
                and pd.notna(representative_slope)
                and pd.notna(tangent_intercept)
                and representative_slope != 0
            ):
                x_intersection = float(-tangent_intercept / representative_slope)

                if pd.notna(tangent_time_s):
                    data_before_tangent = sample_data[
                        sample_data[age_col] <= tangent_time_s
                    ]
                    if len(data_before_tangent) > 0:
                        min_value_before_tangent = float(
                            data_before_tangent[target_col].min()
                        )
                        x_intersection_dormant = float(
                            (min_value_before_tangent - tangent_intercept)
                            / representative_slope
                        )

                x_values = sample_data[age_col].to_numpy(dtype=float)
                j_values = sample_data["normalized_heat_j_g"].to_numpy(dtype=float)
                if np.isfinite(x_intersection):
                    x_intersection_j_g = float(np.interp(x_intersection, x_values, j_values))
                if np.isfinite(x_intersection_dormant):
                    x_intersection_dormant_j_g = float(
                        np.interp(x_intersection_dormant, x_values, j_values)
                    )

            onset_time = (
                x_intersection_dormant
                if np.isfinite(x_intersection_dormant)
                else (
                    x_intersection if np.isfinite(x_intersection) else tangent_time_s
                )
            )

            result_data = {
                "sample": sample,
                "sample_short": sample_short,
                "fraction_of_max_for_first_ascending_slope": row.get("fraction_of_max"),
                "range_method_for_first_ascending_slope": row.get(
                    "first_ascending_range_method"
                ),
                "delta_y_w_g_for_first_ascending_slope": row.get(
                    "first_ascending_delta_y_w_g"
                ),
                "flexible_for_first_ascending_slope": row.get(
                    "first_ascending_flexible"
                ),
                "delta_y_multiplier_for_first_ascending_slope": row.get(
                    "first_ascending_delta_y_multiplier"
                ),
                "delta_y_effective_w_g_for_first_ascending_slope": row.get(
                    "first_ascending_delta_y_effective_w_g"
                ),
                "normalized_heat_flow_w_g_threshold_for_first_ascending_slope": row.get(
                    "fraction_threshold_value"
                ),
                "threshold_basis_for_first_ascending_slope": row.get(
                    "fraction_threshold_basis"
                ),
                "gradient_of_first_ascending_slope_to_fraction_of_max": row.get(
                    "first_ascending_slope"
                ),
                "first_ascending_mean_slope_time_s": row.get(
                    "first_ascending_tangent_time_s"
                ),
                "normalized_heat_flow_w_g_at_first_ascending_mean_slope": row.get(
                    "first_ascending_tangent_value"
                ),
                "first_ascending_slope_start_time_s": row.get(
                    "first_ascending_start_time_s"
                ),
                "first_ascending_slope_end_time_s": row.get("first_ascending_end_time_s"),
                "normalized_heat_flow_w_g_at_first_ascending_slope_start": row.get(
                    "first_ascending_start_value"
                ),
                "normalized_heat_flow_w_g_at_first_ascending_slope_end": row.get(
                    "first_ascending_end_value"
                ),
                "number_of_points_for_first_ascending_slope": row.get(
                    "first_ascending_n_points"
                ),
                "number_of_windows_for_first_ascending_mean_slope": row.get(
                    "first_ascending_n_windows"
                ),
                "standard_deviation_for_first_ascending_mean_slope": row.get(
                    "first_ascending_slope_std"
                ),
                "fraction_start_for_first_ascending_mean_slope": row.get(
                    "first_ascending_fraction_start"
                ),
                "fraction_end_for_first_ascending_mean_slope": row.get(
                    "first_ascending_fraction_end"
                ),
                "window_size_for_first_ascending_mean_slope": row.get(
                    "first_ascending_window_size"
                ),
                "first_ascending_window_start_time_s": row.get(
                    "first_ascending_window_start_time_s"
                ),
                "first_ascending_window_end_time_s": row.get(
                    "first_ascending_window_end_time_s"
                ),
                "first_ascending_window_start_value": row.get(
                    "first_ascending_window_start_value"
                ),
                "first_ascending_window_end_value": row.get(
                    "first_ascending_window_end_value"
                ),
                "onset_time_s_from_first_ascending_slope": onset_time,
                "onset_time_min_from_first_ascending_slope": (
                    onset_time / 60 if pd.notna(onset_time) else None
                ),
                "onset_time_s_from_first_ascending_slope_abscissa": (
                    x_intersection if np.isfinite(x_intersection) else None
                ),
                "normalized_heat_at_onset_time_first_ascending_slope_abscissa_j_g": (
                    x_intersection_j_g if np.isfinite(x_intersection_j_g) else None
                ),
                "normalized_heat_at_onset_time_first_ascending_slope_dormant_j_g": (
                    x_intersection_dormant_j_g
                    if np.isfinite(x_intersection_dormant_j_g)
                    else None
                ),
                "min_value_before_first_ascending_tangent": (
                    min_value_before_tangent if np.isfinite(min_value_before_tangent) else None
                ),
                "peak_time_s": peak_time_s if np.isfinite(peak_time_s) else None,
                "normalized_heat_flow_w_g_at_peak": (
                    peak_value if np.isfinite(peak_value) else None
                ),
                "normalized_heat_j_g_at_peak": peak_j_g if np.isfinite(peak_j_g) else None,
            }
            results.append(result_data)

        return pd.DataFrame(results)

    def _calculate_max_slope_analysis(
        self,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
    ) -> pd.DataFrame:
        """Calculate maximum slope analysis and return structured results."""
        # Get required data
        max_slope_analyzer = SlopeAnalyzer(params)
        max_slopes = max_slope_analyzer.get_maximum_slope(
            self._data,
            target_col,
            age_col,
            regex,
        )

        if max_slopes.empty:
            logger.warning("No maximum slopes found. Check processing parameters.")
            return pd.DataFrame()

        dormant_hfs = self.get_dormant_period_heatflow(params, regex, show_plot=False)
        if dormant_hfs.empty:
            logger.warning("No dormant period heat flows found.")
            return pd.DataFrame()

        # Calculate onsets
        analyzer = OnsetAnalyzer(params)
        onsets = analyzer.get_peak_onset_via_max_slope(
            self._data,
            max_slopes,
            dormant_hfs,  # intersection, xunit
        )

        # Structure results with consistent naming
        results = []
        for _, slope_row in max_slopes.iterrows():
            sample = slope_row.get("sample", slope_row.get("sample_short", ""))
            sample_short = slope_row.get("sample_short", slope_row.get("sample", ""))

            onset_row = (
                onsets[onsets["sample_short"] == sample_short]
                if not onsets.empty
                else pd.DataFrame()
            )
            onset_time = (
                onset_row.iloc[0]["onset_time_s"] if not onset_row.empty else None
            )

            # get normalized_heat_j_g at onset_time
            if onset_time and not pd.isna(onset_time):
                onset_j_g = np.interp(
                    onset_time,
                    self._data[age_col],
                    self._data["normalized_heat_j_g"],
                )
            else:
                onset_j_g = None

            result_data = {
                "sample": sample,
                "sample_short": sample_short,
                "gradient_from_max_slope": slope_row.get("gradient", 0),
                "curvature_at_max_slope": slope_row.get("curvature", 0),
                "max_slope_time_s": slope_row.get("time_s", 0),
                "normalized_heat_flow_w_g_at_max_slope": slope_row.get(
                    "normalized_heat_flow_w_g", 0
                ),
                "normalized_heat_j_g_at_max_slope": slope_row.get("normalized_heat_j_g", 0),
                "normalized_heat_j_g_at_onset_time_max_slope": onset_j_g,
                "onset_time_s_from_max_slope": onset_time,
                "onset_time_min_max_slope": onset_time / 60 if onset_time else None,
                "onset_time_s_max_slope_abscissa": (
                    onset_row.iloc[0]["onset_time_s_abscissa"]
                    if not onset_row.empty
                    else None
                ),
            }
            results.append(result_data)

        return pd.DataFrame(results)

    def _calculate_mean_slope_analysis(
        self,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
    ) -> pd.DataFrame:
        """Calculate mean slope (flank tangent) analysis and return structured results."""
        analyzer = FlankTangentAnalyzer(params)

        # Get flank tangent results
        tangent_results = analyzer.get_ascending_flank_tangent(
            self._data,
            target_col,
            age_col,
            regex,
        )

        if tangent_results.empty:
            logger.warning("No flank tangent results found.")
            return pd.DataFrame()

        results = []
        for _, row in tangent_results.iterrows():
            sample = row.get("sample", row.get("sample_short", ""))
            sample_short = row.get("sample_short", row.get("sample", ""))

            # onset by intersection with tangent to dormant period
            onset_time = row.get("x_intersection_dormant", row.get("tangent_time_s", 0))

            result_data = {
                "sample": sample,
                "sample_short": sample_short,
                "gradient_of_mean_slope": row.get("tangent_slope", 0),
                "mean_slope_time_s": row.get("tangent_time_s", 0),
                "normalized_heat_flow_w_g_at_mean_slope": row.get("tangent_value", 0),
                "normalized_heat_j_g_at_mean_slope": row.get("tangent_j_g", 0),
                "onset_time_s_from_mean_slope": onset_time,
                "onset_time_min_from_mean_slope": onset_time / 60 if onset_time else None,
                "onset_time_s_from_mean_slope_abscissa": row.get("x_intersection", 0),
                "normalized_heat_at_onset_time_mean_slope_abscissa_j_g": row.get("x_intersection_j_g", 0),
                "normalized_heat_at_onset_time_mean_slope_dormant_j_g": row.get("x_intersection_dormant_j_g", 0),
                "flank_start_value": row.get("flank_start_value", 0),
                "flank_end_value": row.get("flank_end_value", 0),
                "peak_time_s": row.get("peak_time_s", 0),
                "normalized_heat_flow_w_g_at_peak": row.get("peak_value", 0),
                "normalized_heat_j_g_at_peak": row.get("peak_j_g", 0),
            }
            results.append(result_data)

        return pd.DataFrame(results)

    def _merge_slope_results(
        self,
        max_slope_results: pd.DataFrame,
        mean_slope_results: pd.DataFrame,
        dormant_hf_results: pd.DataFrame,
        astm_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge max slope and mean slope results into comprehensive DataFrame."""
        frames = [
            frame
            for frame in [
                max_slope_results,
                mean_slope_results,
                dormant_hf_results,
                astm_results,
            ]
            if not frame.empty
        ]

        if not frames:
            return pd.DataFrame()

        combined = frames[0]
        for frame in frames[1:]:
            combined = pd.merge(
                combined,
                frame,
                on=["sample", "sample_short"],
                how="outer",
                suffixes=("", "_duplicate"),
            )

        duplicate_cols = [col for col in combined.columns if col.endswith("_duplicate")]
        combined = combined.drop(columns=duplicate_cols)

        return combined

    def _plot_combined_slope_analysis(
        self,
        results: pd.DataFrame,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        plot_type: str,
        regex: Optional[str],
        plotpath: Optional[pathlib.Path],
        ax,
        show_plot: bool = True,
    ):
        """
        Plot combined slope analysis results based on plot_type parameter.

        Parameters
        ----------
        results : pd.DataFrame
            Combined results containing both max and mean slope data
        target_col : str
            Column name for heat flow data
        age_col : str
            Column name for time data
        plot_type : str
            Type of plot to show: 'max', 'mean', or 'both'
            - 'max': Shows only maximum slope analysis plot
            - 'mean': Shows only mean slope (flank tangent) analysis plot
            - 'both': Shows both analysis types (separate plots for each)
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        cutoff_min : float, optional
            Cutoff time in minutes
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        show_plot : bool, optional
            Whether to show the plot, by default True
        """
        # Validate plot_type parameter
        valid_plot_types = ["max", "mean", "both"]
        cutoff_min = params.cutoff.cutoff_min

        if plot_type not in valid_plot_types:
            raise ValueError(
                f"plot_type must be one of {valid_plot_types}, got '{plot_type}'"
            )

        # For now, plot using the existing unified plotting approach
        # This could be enhanced to show both slope methods simultaneously
        for _, result_row in results.iterrows():
            sample = result_row["sample"]
            sample_short = result_row["sample_short"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, cutoff_time_min=cutoff_min
            )
            if sample_data.empty:
                continue

            if not pd.isna(
                result_row.onset_time_s_from_mean_slope or result_row.onset_time_s_from_max_slope
            ):
                self._plotter.plot_tangent_analysis(
                    sample_data,
                    sample_short,
                    params,
                    ax=ax,
                    age_col=age_col,
                    target_col=target_col,
                    cutoff_time_min=cutoff_min,
                    analysis_type=plot_type,  # Use correct analysis type
                    results=result_row.to_frame().T,
                    figsize=(7, 5),
                    metadata=self._metadata,
                    metadata_id=self._metadata_id,
                )
            self._save_and_show_plot(
                plotpath, f"{plot_type}_slope_{sample_short}.png", ax, show_plot=show_plot
            )

    def _plot_ascending_slope_analysis(
        self,
        results: pd.DataFrame,
        params: ProcessingParameters,
        target_col: str,
        age_col: str,
        regex: Optional[str],
        plotpath: Optional[pathlib.Path],
        ax,
        show_plot: bool = True,
    ):
        """Plot first ascending slope analysis results."""
        cutoff_min = params.cutoff.cutoff_min

        for _, result_row in results.iterrows():
            sample = result_row["sample"]
            sample_short = result_row["sample_short"]

            sample_data = self._get_filtered_sample_data(
                sample, age_col, cutoff_time_min=cutoff_min
            )
            if sample_data.empty:
                continue

            self._plotter.plot_tangent_analysis(
                sample_data,
                sample_short,
                params,
                ax=ax,
                age_col=age_col,
                target_col=target_col,
                cutoff_time_min=cutoff_min,
                analysis_type="ascending",
                results=result_row.to_frame().T,
                figsize=(7, 5),
                metadata=self._metadata,
                metadata_id=self._metadata_id,
            )

            self._save_and_show_plot(
                plotpath,
                f"ascending_slope_{sample_short}.png",
                ax,
                show_plot=show_plot,
            )


    # Backward compatibility methods
    def get_peak_onset_via_max_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        show_plot: bool = False,
        ax=None,
        regex: Optional[str] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        time_discarded_s: float = 900,
        save_path: Optional[pathlib.Path] = None,
        xscale: str = "linear",
        xunit: str = "s",
        intersection: str = "dormant_hf",
    ):
        """
        Get reaction onset via maximum slope intersection method.

        This is a wrapper around get_peak_onset_via_slope for backward compatibility.
        Returns only the max slope related columns for compatibility.
        """
        full_results = self.get_mainpeak_params(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            show_plot=show_plot,
            regex=regex,
            ax=ax,
            plot_type="max",

            #time_discarded_s=time_discarded_s,
            #intersection=intersection,
            #xunit=xunit,
        )

        if full_results.empty:
            return full_results

        # Extract only max slope related columns for backward compatibility
        # max_slope_cols = [
        #     col
        #     for col in full_results.columns
        #     if col.startswith("max_slope_") or col in ["sample", "sample_short"]
        # ]

        # result = full_results[max_slope_cols].copy()

        # Rename columns to match old API
        # column_mapping = {
        #     "onset_time_s_from_max_slope": "onset_time_s",
        #     "max_slope_onset_time_min": "onset_time_min",
        #     "max_slope_value": "maximum_slope",
        #     "max_slope_time_s": "maximum_slope_time_s",
        # }

        # for old_name, new_name in column_mapping.items():
        #     if old_name in result.columns:
        #         result = result.rename(columns={old_name: new_name})

        return full_results
    

    def get_ascending_flank_tangent(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        flank_fraction_start: float = 0.2,
        flank_fraction_end: float = 0.8,
        window_size: float = 0.1,
        cutoff_min: Optional[float] = None,
        show_plot: bool = False,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        ax=None,
    ) -> pd.DataFrame:
        """
        Determine tangent to ascending flank of peak by averaging over sections.

        This is a wrapper around get_peak_onset_via_slope for backward compatibility.
        Returns only the mean slope related columns for compatibility.
        """
        full_results = self.get_peak_onset_via_slope(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            cutoff_min=cutoff_min,
            show_plot=show_plot,
            regex=regex,
            plotpath=plotpath,
            ax=ax,
            flank_fraction_start=flank_fraction_start,
            flank_fraction_end=flank_fraction_end,
            window_size=window_size,
        )

        if full_results.empty:
            return full_results

        # Extract only mean slope related columns for backward compatibility
        mean_slope_cols = [
            col
            for col in full_results.columns
            if col.startswith("mean_slope_")
            or col in ["sample", "sample_short", "peak_time_s", "peak_value"]
        ]

        result = full_results[mean_slope_cols].copy()

        # Rename columns to match old API
        column_mapping = {
            "mean_slope_onset_time_s": "x_intersection",
            "mean_slope_value": "tangent_slope",
            "mean_slope_time_s": "tangent_time_s",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in result.columns:
                result = result.rename(columns={old_name: new_name})

        return result

    def get_dormant_period_heatflow(
        self,
        processparams: Optional[ProcessingParameters] = None,
        regex: Optional[str] = None,
        cutoff_min: int = 5,
        upper_dormant_thresh_w_g: float = 0.002,
        plot_right_boundary: float = 2e5,
        prominence: float = 1e-3,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        """Get dormant period heat flow characteristics."""
        params = processparams or self.processparams

        # Get peaks first
        peaks = self.get_peaks(params, regex=regex, show_plot=False)

        # Analyze dormant period
        analyzer = DormantPeriodAnalyzer(params)
        dorm_hf = analyzer.get_dormant_period_heatflow(
            self._data, peaks, regex, upper_dormant_thresh_w_g
        )

        if not dorm_hf.empty:
            return dorm_hf
        else:
            return pd.DataFrame()

    def get_astm_c1679_characteristics(
        self,
        processparams: Optional[ProcessingParameters] = None,
        individual: bool = True,
        show_plot: bool = False,
        ax=None,
        regex: Optional[str] = None,
        xscale: str = "log",
        xunit: str = "s",
    ) -> pd.DataFrame:
        """Get characteristics according to ASTM C1679."""
        params = processparams or self.processparams

        peaks = self.get_peaks(params, regex=regex, show_plot=False)

        analyzer = ASTMC1679Analyzer(params)
        df = analyzer.get_astm_c1679_characteristics(
            self._data, peaks, individual, regex
        )

        if show_plot and not df.empty:
            self._plotter.plot_astm_c1679(self._data, df, ax, xunit)

        return df

    def get_cumulated_heat_at_hours(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_h: float = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """Get cumulated heat flow at specific age."""
        if "cutoff_min" in kwargs:
            cutoff_min = kwargs["cutoff_min"]
            warnings.warn(
                "The cutoff_min parameter is deprecated. Use ProcessingParameters instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            params = processparams or self.processparams
            cutoff_min = params.cutoff.cutoff_min

        return HeatCalculator.get_cumulated_heat_at_hours(
            self._data, target_h, cutoff_min
        )

    def get_average_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
        show_plot: bool = False,
        ax=None,
        save_path: Optional[pathlib.Path] = None,
        xunit: str = "s",
    ) -> pd.DataFrame:
        """Calculate the mean (flank tangent) slope of the main hydration peak.

        Wrapper around :meth:`get_mainpeak_params` returning only the
        ``mean_slope`` columns together with sample identifiers.
        """
        result = self.get_mainpeak_params(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            show_plot=show_plot,
            plot_type="mean",
            regex=regex,
            ax=ax,
        )
        if result.empty:
            return result
        id_cols = ["sample", "sample_short"]
        mean_slope_cols = [c for c in result.columns if "mean_slope" in c]
        return result[id_cols + mean_slope_cols].reset_index(drop=True)

    def _plot_tangent_analysis_unified(
        self,
        results: pd.DataFrame,
        analysis_type: str,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
        # Additional data for onset intersection analysis
        max_slopes: Optional[pd.DataFrame] = None,
        dormant_hfs: Optional[pd.DataFrame] = None,
        onsets: Optional[pd.DataFrame] = None,
    ):
        """
        Unified plotting method for tangent-based analysis results.

        This method handles both flank tangent and onset intersection analysis,
        with the main difference being how the slope is determined:
        - Flank tangent: Uses averaged slope over a window
        - Max slope: Uses single point with maximum gradient

        Parameters
        ----------
        results : pd.DataFrame
            Results from the analysis (tangent results for flank, onsets for max slope)
        analysis_type : str
            Either 'flank_tangent' or 'max_slope_onset'
        target_col : str
            Column name for heat flow data
        age_col : str
            Column name for time data
        regex : str, optional
            Regex to filter samples
        plotpath : pathlib.Path, optional
            Path to save plots
        cutoff_time_min : float, optional
            Cutoff time in minutes
        intersection : str
            Type of intersection for onset analysis ('dormant_hf' or 'abscissa')
        xunit : str
            Time unit for plotting
        time_discarded_s : float
            Time to discard for onset analysis
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        max_slopes : pd.DataFrame, optional
            Required for onset intersection analysis
        dormant_hfs : pd.DataFrame, optional
            Required for onset intersection analysis with dormant_hf
        onsets : pd.DataFrame, optional
            Required for onset intersection analysis
        """
        try:
            if analysis_type == "flank_tangent":
                self._plot_flank_tangent_unified(
                    results, target_col, age_col, regex, plotpath, cutoff_time_min, ax
                )
            elif analysis_type == "max_slope_onset":
                self._plot_onset_intersection_unified(
                    results,
                    max_slopes,
                    dormant_hfs,
                    target_col,
                    age_col,
                    regex,
                    intersection,
                    xunit,
                    time_discarded_s,
                    ax,
                )
            else:
                raise ValueError(f"Unknown analysis_type: {analysis_type}")

        except Exception as e:
            logger.error(f"Error plotting tangent analysis results: {e}")
            print(f"Plotting failed: {e}")

    def _plot_flank_tangent_unified(
        self,
        results: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        ax=None,
    ):
        """Plot flank tangent analysis results using unified SimplePlotter."""
        for _, result_row in results.iterrows():
            sample = result_row["sample"]
            sample_short = result_row["sample_short"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, cutoff_time_min=cutoff_time_min
            )
            if sample_data.empty:
                continue

            # Create a DataFrame with just this result for plotting
            single_result = pd.DataFrame([result_row])

            # Use unified plotting method
            self._plotter.plot_tangent_analysis(
                sample_data,
                sample_short,
                ax=ax,
                age_col=age_col,
                target_col=target_col,
                cutoff_time_min=cutoff_time_min,
                analysis_type="flank_tangent",
                tangent_results=single_result,
                figsize=(7, 5),
            )

            self._save_and_show_plot(plotpath, f"flank_tangent_{sample_short}.png", ax)

    def _plot_onset_intersection_unified(
        self,
        onsets: pd.DataFrame,
        max_slopes: Optional[pd.DataFrame],
        dormant_hfs: Optional[pd.DataFrame],
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
    ):
        """Plot onset intersection analysis results using unified SimplePlotter."""
        if max_slopes is None:
            raise ValueError("max_slopes required for onset intersection analysis")

        for _, onset_row in onsets.iterrows():
            sample = onset_row["sample"]

            # Get sample data
            sample_data = self._get_filtered_sample_data(
                sample, age_col, time_discarded_s=time_discarded_s
            )
            if sample_data.empty:
                continue

            # Use unified plotting method
            self._plotter.plot_tangent_analysis(
                sample_data,
                sample,
                ax=ax,
                age_col=age_col,
                target_col=target_col,
                analysis_type="onset_intersection",
                max_slopes=max_slopes,
                dormant_hfs=dormant_hfs,
                onsets=onsets,
                intersection=intersection,
                xunit=xunit,
                figsize=(12, 8),
            )

            # Note: plotpath not available in this context, only show plot
            self._save_and_show_plot(None, f"onset_intersection_{sample}.png", ax)

    def _get_filtered_sample_data(
        self,
        sample: str,
        age_col: str,
        cutoff_time_min: Optional[float] = None,
        time_discarded_s: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get sample data with appropriate filtering applied.

        This consolidates the common data filtering logic used in both analysis types.
        """
        # Get sample data - handle both 'sample' and 'sample_short' columns
        sample_data = self._data[
            (self._data["sample"] == sample)
            | (self._data.get("sample_short", "") == sample)
        ]

        if sample_data.empty:
            return sample_data

        # Apply cutoff time filtering
        if cutoff_time_min is not None:
            cutoff_seconds = cutoff_time_min * 60
            sample_data = sample_data[sample_data[age_col] >= cutoff_seconds]

        # Apply time discarded filtering (for onset analysis)
        if time_discarded_s is not None and time_discarded_s > 0:
            sample_data = sample_data[sample_data[age_col] >= time_discarded_s]

        return sample_data

    def _save_and_show_plot(self, plotpath: Optional[pathlib.Path], filename: str, ax, show_plot: bool = True):
        """Handle plot saving and showing - common logic for both analysis types."""
        import matplotlib.pyplot as plt

        if plotpath:
            plot_file = plotpath / filename
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")

        if not ax:
            if show_plot:
                plt.show()
            else:
                plt.close()

    def _plot_flank_tangent_results(
        self,
        results: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        cutoff_time_min: Optional[float] = None,
        ax=None,
    ):
        """
        Plot flank tangent analysis results using SimplePlotter.

        This is a wrapper around the unified plotting method for backward compatibility.
        """
        return self._plot_tangent_analysis_unified(
            results=results,
            analysis_type="flank_tangent",
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            plotpath=plotpath,
            cutoff_time_min=cutoff_time_min,
            ax=ax,
        )

    def _plot_onset_intersections(
        self,
        onsets: pd.DataFrame,
        max_slopes: pd.DataFrame,
        dormant_hfs: pd.DataFrame,
        target_col: str,
        age_col: str,
        regex: Optional[str] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        time_discarded_s: float = 900,
        ax=None,
    ):
        """
        Plot onset intersection analysis results using SimplePlotter.

        This is a wrapper around the unified plotting method for backward compatibility.
        """
        return self._plot_tangent_analysis_unified(
            results=onsets,
            analysis_type="max_slope_onset",
            target_col=target_col,
            age_col=age_col,
            regex=regex,
            intersection=intersection,
            xunit=xunit,
            time_discarded_s=time_discarded_s,
            ax=ax,
            max_slopes=max_slopes,
            dormant_hfs=dormant_hfs,
            onsets=onsets,
        )

    # Data manipulation methods
    def normalize_sample_to_mass(
        self, sample_short: str, mass_g: float, show_info: bool = True
    ):
        """Normalize heat flow values to a specific mass."""
        self._data = DataNormalizer.normalize_sample_to_mass(
            self._data, sample_short, mass_g, show_info
        )

    def apply_tian_correction(
        self, processparams: Optional[ProcessingParameters] = None
    ) -> None:
        """
        Apply Tian correction to the heat flow data.

        Corrects the measured heat flow for the thermal inertia of the calorimeter
        using one or two time constants (tau1, tau2) from processparams.

        Single time constant (tau2 = None):
            hf_corrected = dHF/dt * tau1 + HF

        Dual time constants:
            hf_corrected = dHF/dt * (tau1 + tau2) + d²HF/dt² * tau1*tau2 + HF

        Results are written to three new columns:
        - ``normalized_heat_flow_w_g_tian``
        - ``gradient_normalized_heat_flow_w_g``
        - ``normalized_heat_j_g_tian``

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters containing time_constants.tau1 and
            time_constants.tau2. Uses instance processparams if not provided.
        """
        from scipy import integrate

        if processparams is None:
            processparams = self._processparams

        for s, sample_data in SampleIterator.iter_samples(self._data):
            processor = HeatFlowProcessor(processparams)
            gradient, curvature = processor.calculate_heatflow_derivatives(sample_data)

            hf = sample_data["normalized_heat_flow_w_g"].to_numpy()
            x = sample_data["time_s"].to_numpy()
            tau1 = processparams.time_constants.tau1

            if processparams.time_constants.tau2 is None:
                norm_hf = gradient * tau1 + hf
            else:
                tau2 = processparams.time_constants.tau2
                norm_hf = gradient * (tau1 + tau2) + curvature * tau1 * tau2 + hf

            mask = self._data["sample"] == s
            self._data.loc[mask, "normalized_heat_flow_w_g_tian"] = norm_hf
            self._data.loc[mask, "gradient_normalized_heat_flow_w_g"] = gradient
            self._data.loc[mask, "normalized_heat_j_g_tian"] = (
                integrate.cumulative_trapezoid(
                    np.nan_to_num(norm_hf), x=x, initial=0
                )
            )

    def add_metadata_source(
        self,
        file: Union[str, pathlib.Path],
        sample_id_column: str,
        show_info: bool = True,
    ):
        """
        Add metadata from external source (CSV or Excel file).

        Parameters
        ----------
        file : str or pathlib.Path
            Path to metadata file (CSV, Excel, etc.)
        sample_id_column : str
            Column name in metadata file that matches sample names
        show_info : bool, optional
            Whether to print informative messages, by default True

        Raises
        ------
        FileNotFoundError
            If the metadata file does not exist
        ValueError
            If the sample_id_column is not found in the metadata file
        """
        file_path = pathlib.Path(file)

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        # Load metadata based on file extension
        try:
            if file_path.suffix.lower() in [".xlsx", ".xls"]:
                self._metadata = pd.read_excel(file_path)
            elif file_path.suffix.lower() == ".csv":
                self._metadata = pd.read_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. Use CSV or Excel files."
                )

            if show_info:
                print(f"Loaded metadata from: {file_path}")
                print(f"Metadata shape: {self._metadata.shape}")

        except Exception as e:
            logger.error(f"Error loading metadata file: {e}")
            raise ValueError(f"Failed to load metadata from {file_path}: {e}")

        # Validate sample_id_column
        if sample_id_column not in self._metadata.columns:
            raise ValueError(
                f"Column '{sample_id_column}' not found in metadata. "
                f"Available columns: {list(self._metadata.columns)}"
            )

        self._metadata_id = sample_id_column

        # Try to match metadata with existing samples
        if not self._data.empty and "sample_short" in self._data.columns:
            sample_names = self.get_sample_names()
            metadata_ids = self._metadata[sample_id_column].unique()

            matched = set(sample_names) & set(metadata_ids)
            unmatched_samples = set(sample_names) - matched
            unmatched_metadata = set(metadata_ids) - matched

            if show_info:
                print("\nMetadata matching results:")
                print(f"  Matched samples: {len(matched)}")
                if unmatched_samples:
                    print(f"  Unmatched samples: {len(unmatched_samples)}")
                    print(f"    {list(unmatched_samples)[:5]}{'...' if len(unmatched_samples) > 5 else ''}")
                if unmatched_metadata:
                    print(f"  Unmatched metadata entries: {len(unmatched_metadata)}")

        if show_info:
            print(f"Metadata successfully added with ID column: '{sample_id_column}'")

    def average_by_metadata(
        self,
        groupby: str | list[str],
        bin_width_s: int = 60,
    ) -> None:
        """Replace individual samples with group averages defined by metadata.

        Requires :meth:`add_metadata_source` to have been called first.
        The averaged data replaces ``self._data`` in-place so that all
        downstream methods (``plot``, ``get_cumulated_heat_at_hours``, …)
        operate on the grouped curves.  Call :meth:`undo_average_by_metadata`
        to restore the original data.

        Parameters
        ----------
        groupby : str or list of str
            Metadata column(s) to group by, e.g. ``"cement_name"`` or
            ``["cement_name", "cement_amount_g"]``.
        bin_width_s : int
            Width of each time bin in seconds. Default is 60 s.
        """
        if self._metadata.empty:
            raise ValueError(
                "No metadata loaded. Call add_metadata_source() first."
            )
        self._data_before_average = self._data.copy()
        self._data = MetadataAggregator.average_by_metadata(
            self._data,
            self._metadata,
            self._metadata_id,
            groupby,
            bin_width_s,
        )

    def undo_average_by_metadata(self) -> None:
        """Restore the original per-sample data after average_by_metadata."""
        if not hasattr(self, "_data_before_average"):
            raise ValueError(
                "No averaged data to undo. Call average_by_metadata() first."
            )
        self._data = self._data_before_average
        del self._data_before_average

    def remove_pickle_files(self):
        """Remove pickle cache files."""
        self._data_persistence.remove_pickle_files()

    # Private utility methods
    def _iter_samples(self, regex: Optional[str] = None):
        """Iterate over samples - compatibility method."""
        return SampleIterator.iter_samples(self._data, regex)
