"""
Refactored main measurement class for calorimetry data handling.
"""

import copy
import logging
import pathlib
import re
import warnings
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analysis import (
    ASTMC1679Analyzer,
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
from .file_io import (
    DataPersistence,
    FolderDataLoader,
    load_measurement,
    save_measurement,
)
from .plotting import SimplePlotter
from .processparams import ProcessingParameters
from .utils import adaptive_downsample, downsample_sections

logger = logging.getLogger(__name__)


def _warn_ignored_args(method: str, **kwargs) -> None:
    """Emit a DeprecationWarning for each argument that was passed but has no effect."""
    for name, value in kwargs.items():
        if value is not None and value is not False:
            warnings.warn(
                f"{method}(): the '{name}' parameter has no effect and will be "
                "removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )


class Measurement:
    """
    Isothermal heat flow calorimetry data of one or more samples.

    A Measurement reads all calorimetry files of a folder into one long-format
    DataFrame and provides methods for plotting and analysing the samples.

    Parameters
    ----------
    folder : str or pathlib.Path, optional
        Folder with the exported calorimetry files (``.csv`` or ``.xls``).
        Subfolders are not searched. If None, an empty Measurement is created.
    show_info : bool, optional
        Print information while loading, by default True.
    regex : str, optional
        Regular expression matched against the beginning of the file names;
        only matching files are loaded.
    auto_clean : bool, optional
        Drop rows without heat flow values and merge differently named
        temperature columns into ``temperature_c``, by default False.
    cold_start : bool, optional
        If True (default), read the files. If False, read ``_data.pickle`` and
        ``_info.pickle`` from the current working directory instead (see
        ``save_cache``).
    processparams : ProcessingParameters, optional
        Default processing parameters for all analysis methods. The ``downsample``
        settings are applied while loading.
    new_code : bool, optional
        Deprecated, has no effect.
    processed : bool, optional
        Read plain CSV files previously written from :meth:`get_data`, by default
        False.
    metadata_path : str or pathlib.Path, optional
        CSV or Excel file with one row of metadata per sample, see
        :meth:`add_metadata_source`.
    metadata_id_column : str, optional
        Column of the metadata file holding the sample names. Required together
        with ``metadata_path``.
    save_cache : bool, optional
        Write ``_data.pickle`` and ``_info.pickle`` to the current working
        directory after loading, by default False. Prefer :meth:`save`.

    Raises
    ------
    AutoCleanException
        If ``auto_clean`` fails.
    ColdStartException
        If ``cold_start=False`` and no cache files are found.

    Examples
    --------
    >>> from calocem import Measurement, ProcessingParameters
    >>> tam = Measurement(folder="calo_data", regex=r"opc_.*", show_info=False)
    >>> tam.get_sample_names()
    ['opc_1', 'opc_2']
    """

    def __init__(
        self,
        folder: Optional[Union[str, pathlib.Path]] = None,
        show_info: bool = True,
        regex: Optional[str] = None,
        auto_clean: bool = False,
        cold_start: bool = True,
        processparams: Optional[ProcessingParameters] = None,
        new_code: Optional[bool] = None,
        processed: bool = False,
        metadata_path: Optional[Union[str, pathlib.Path]] = None,
        metadata_id_column: Optional[str] = None,
        save_cache: bool = False,
    ):
        # Initialize attributes
        self._data = pd.DataFrame()
        self._info = pd.DataFrame()
        self._data_unprocessed = pd.DataFrame()
        self._metadata = pd.DataFrame()
        self._metadata_id = ""
        self._average_groupby: Optional[list[str]] = None

        _warn_ignored_args("Measurement", new_code=new_code)

        # Store configuration
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
                    raise AutoCleanException() from e
                if not cold_start:
                    raise ColdStartException() from e
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
            raise DataProcessingException("load_from_folder", e) from e

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
        """Apply adaptive downsampling to the heat flow data, per sample."""
        downsampled = []
        for _, sample_data in SampleIterator.iter_samples(self._data):
            sample_data = sample_data.dropna(subset=["normalized_heat_flow_w_g"])
            if sample_data.empty:
                continue

            if self.processparams.downsample.section_split:
                sample_data = downsample_sections(
                    sample_data,
                    x_col="time_s",
                    y_col="normalized_heat_flow_w_g",
                    processparams=self.processparams,
                )
            else:
                sample_data = adaptive_downsample(
                    sample_data,
                    x_col="time_s",
                    y_col="normalized_heat_flow_w_g",
                    processparams=self.processparams,
                )
            downsampled.append(sample_data)

        if downsampled:
            self._data = pd.concat(downsampled)

    # Data access methods
    def get_data(self) -> pd.DataFrame:
        """
        Return the calorimetry data of all samples.

        Returns
        -------
        pd.DataFrame
            Long-format data with one row per sample and time. Typical columns are
            ``time_s`` (s), ``temperature_c`` (°C), ``heat_flow_w`` (W), ``heat_j``
            (J), ``normalized_heat_flow_w_g`` (W/g), ``normalized_heat_j_g`` (J/g),
            ``sample`` (path of the source file) and ``sample_short`` (sample name).

        Notes
        -----
        The returned DataFrame is the internal data, not a copy. Use ``.copy()``
        before modifying it.
        """
        return self._data

    def get_information(self) -> pd.DataFrame:
        """
        Return the sample information stored in the calorimetry files.

        Returns
        -------
        pd.DataFrame
            Information such as the sample mass, with ``sample`` and ``sample_short``
            columns. The content depends on the file format.
        """
        return self._info

    def get_metadata(self) -> tuple[pd.DataFrame, str]:
        """
        Return the metadata added with :meth:`add_metadata_source`.

        Returns
        -------
        tuple of (pd.DataFrame, str)
            The metadata table and the name of its sample ID column. Both are empty
            if no metadata was added.
        """
        return self._metadata, self._metadata_id

    def get_data_with_metadata(self) -> pd.DataFrame:
        """Return the data joined with its metadata as one tidy DataFrame.

        Works on per-sample data and after :meth:`average_by_metadata` alike.
        For averaged data, only metadata that is constant within each group is
        attached. The data is returned unchanged when no metadata has been
        added via :meth:`add_metadata_source`.

        Export with, e.g.,
        ``m.get_data_with_metadata().to_csv(path, index=False)``.
        """
        return MetadataAggregator.combine_with_metadata(
            self._data,
            self._metadata,
            self._metadata_id,
            self._average_groupby,
        )

    def get_sample_names(self) -> list[str]:
        """
        Return the names of all loaded samples.

        Returns
        -------
        list of str
            Sample names, i.e. the file names without extension. After
            :meth:`average_by_metadata` the group names are returned.
        """
        return [
            pathlib.Path(str(sample)).stem
            for sample, _ in SampleIterator.iter_samples(self._data)
        ]

    # Persistence methods
    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save this Measurement to a named pickle file for fast reloading.

        The whole object is stored, so the processed (e.g. downsampled) data,
        info, added metadata and ``processparams`` are all preserved and the
        object can be restored with :meth:`load` in a downstream script.

        This file is a convenience cache, not an archive: the raw source files
        remain the source of truth. A file that fails to load -- for instance
        because it was written by an incompatible calocem version -- can always
        be regenerated by re-running from the original folder.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. The ``.pkl`` extension is recommended to make the
            pickle format explicit.
        """
        save_measurement(self, path)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "Measurement":
        """Load a Measurement previously written with :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            File written by :meth:`save`.

        Returns
        -------
        Measurement
            The restored object, ready for plotting and analysis.

        Raises
        ------
        DataProcessingException
            If the file cannot be unpickled, e.g. because it was written by an
            incompatible calocem version. Regenerate it from the raw folder.
        """
        measurement: Measurement = load_measurement(path)
        return measurement

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
        """
        Plot a data column of all samples over time.

        Parameters
        ----------
        t_unit : str, optional
            Time unit of the x-axis: "s", "min", "h" (default) or "d".
        y : str, optional
            Column to plot, by default "normalized_heat_flow_w_g".
        y_unit_milli : bool, optional
            Show heat flow and heat in milli units (mW/g, mJ/g), by default True.
        regex : str, optional
            Plot only samples whose name matches this regular expression.
        show_info : bool, optional
            Print the name of each plotted sample, by default True.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure is created if None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with one line per sample.

        Examples
        --------
        >>> ax = tam.plot(y="normalized_heat_j_g", t_unit="h", y_unit_milli=False)
        >>> ax.set_xlim(0, 48)
        """
        return self._plotter.plot_data(
            self._data, t_unit, y, y_unit_milli, regex, show_info, ax
        )

    def plot_heatflow_with_gradient(
        self,
        processparams: Optional[ProcessingParameters] = None,
        t_unit: str = "h",
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        y_unit_milli: bool = True,
        gradient_unit_milli: bool = True,
        gradient_color: Optional[str] = "orange",
        align_zeros: bool = True,
        show_zero_line: bool = False,
        grid: bool = False,
        regex: Optional[str] = None,
        show_info: bool = True,
        ax=None,
    ):
        """Plot heat flow curve(s) with their gradient on a secondary y-axis.

        The gradient is computed using the smoothing settings (rolling mean,
        median filter, spline interpolation) from ``processparams`` (or the
        instance's processparams when not provided).

        Returns
        -------
        (ax, ax_grad) : tuple of matplotlib.axes.Axes
            Primary (heat flow) and secondary (gradient) axes.
        """
        params = processparams or self.processparams
        return self._plotter.plot_heatflow_with_gradient(
            self._data,
            params,
            t_unit=t_unit,
            target_col=target_col,
            age_col=age_col,
            y_unit_milli=y_unit_milli,
            gradient_unit_milli=gradient_unit_milli,
            gradient_color=gradient_color,
            align_zeros=align_zeros,
            show_zero_line=show_zero_line,
            grid=grid,
            regex=regex,
            show_info=show_info,
            ax=ax,
        )

    def plot_by_category(
        self,
        categories: Union[str, list[str]],
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
    ):
        """
        Plot the samples group-wise, grouped by one or more metadata columns.

        Requires metadata added via ``metadata_path`` or :meth:`add_metadata_source`.

        Parameters
        ----------
        categories : str or list of str
            Metadata column(s) to group by.
        t_unit : str, optional
            Time unit, see :meth:`plot`. By default "h".
        y : str, optional
            Column to plot, see :meth:`plot`. By default "normalized_heat_flow_w_g".
        y_unit_milli : bool, optional
            See :meth:`plot`. By default True.

        Yields
        ------
        tuple
            ``(category_value, ax)`` for each group, where ``category_value`` is a
            tuple when grouping by several columns.

        Examples
        --------
        >>> for value, ax in tam.plot_by_category(["date", "cement_name"]):
        ...     ax.set_ylim(0, 3)
        ...     plt.show()
        """
        if self._metadata.empty:
            raise ValueError(
                "plot_by_category requires metadata; call add_metadata_source first."
            )

        def join_words(values) -> str:
            words = [str(v) for v in values]
            if len(words) > 1:
                return ", ".join(words[:-1]) + " and " + words[-1]
            return words[0] if words else ""

        by = [categories] if isinstance(categories, str) else list(categories)
        loaded = self._metadata[self._metadata_id].astype(str).isin(
            self._data["sample_short"].astype(str)
        )
        for selection, group in self._metadata[loaded].groupby(by=by):
            samples = group[self._metadata_id].astype(str)
            regex = "|".join(f"({re.escape(sample)})" for sample in samples)

            ax = self.plot(
                t_unit=t_unit,
                y=y,
                y_unit_milli=y_unit_milli,
                regex=regex,
                show_info=False,
            )
            ax.set_title(f"Grouped by {join_words(by)} ({join_words(selection)})")

            yield (selection if len(by) > 1 else selection[0]), ax

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
        """
        Detect peaks in the heat flow of each sample.

        Peaks are searched after ``processparams.cutoff.cutoff_min`` with
        :func:`scipy.signal.find_peaks`, using ``processparams.peakdetection``.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        cutoff_min : float, optional
            Deprecated, use ``processparams.cutoff.cutoff_min``.
        show_plot : bool, optional
            Plot each sample with its peaks, by default True.
        plt_right_s, plt_top, xunit, plot_labels, xmarker
            Currently have no effect.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, one figure per sample is created.

        Returns
        -------
        pd.DataFrame
            One row per peak: the data row at the peak (``time_s``,
            ``normalized_heat_flow_w_g``, ...), its ``prominence`` and ``peak_nr``
            (counting from 0 for each sample).
        """
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
        """
        Find the first time at which the heat flow gradient exceeds a threshold.

        Parameters
        ----------
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        time_discarded_s : float, optional
            Ignore data before this time in seconds, by default 900.
        rolling : int, optional
            Window (number of points) of a rolling mean applied before the gradient
            is taken, by default 1 (no smoothing).
        gradient_threshold : float, optional
            Gradient threshold in W/g/s, by default 0.0005.
        show_plot, ax
            Deprecated, have no effect.
        exclude_discarded_time : bool, optional
            Also remove the discarded data before computing the gradient, by default
            False.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.

        Returns
        -------
        pd.DataFrame
            One row per sample with the data row at the onset and its ``gradient``.
            Samples that never exceed the threshold are missing.

        Notes
        -----
        This method does not use the processing parameters. The default threshold is
        too high for typical cement pastes; values around 1e-8 W/g/s are a better
        starting point.
        """
        _warn_ignored_args("get_peak_onsets", show_plot=show_plot, ax=ax)
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
        """
        Fit the heat flow of each sample as a sum of peaks plus a baseline.

        The data after ``processparams.cutoff.cutoff_min`` is fitted with
        ``n_peaks`` lognormal or Gaussian components by constrained least squares.
        If the fit with the requested baseline fails, simpler baselines are tried.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        n_peaks : int, optional
            Number of components. Defaults to ``processparams.deconvolution.n_peaks``.
        peak_shape : str, optional
            "lognormal" (default) or "gaussian".
        baseline_mode : str, optional
            "constant", "linear", "chebyshev" or "none". Defaults to
            ``processparams.deconvolution.baseline_mode``.
        relative_intensity_upper_bounds : list of float, optional
            One value per component in (0, 1]: maximum share of the total amplitude.
        peak_width_upper_bounds : list of float, optional
            One value per component: maximum width.
        show_plot : bool, optional
            Plot the data, the components and the fit, by default False.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.

        Returns
        -------
        pd.DataFrame
            One row per sample and component with ``component`` (from 1),
            ``amplitude`` (W/g), ``center_time_s``, ``peak_time_s``, ``width``,
            ``component_area`` (J/g), ``component_area_fraction``, the fit quality
            ``fit_r2`` and ``fit_rmse`` (W/g) and the baseline coefficients.

        Notes
        -----
        For lognormal components, ``width`` is the dimensionless standard deviation
        of ln(t) and ``center_time_s`` the median; for Gaussian components both are
        in seconds. A component with an amplitude close to zero has a meaningless
        ``peak_time_s``.
        """
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
        """
        Determine an onset from the tangent at the inflection point of the first peak.

        The earliest fitted component of :meth:`get_deconvolution` is taken, and the
        tangent at the inflection point of its ascending flank is intersected with
        the time axis.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        n_peaks, peak_shape, baseline_mode
            Passed to :meth:`get_deconvolution` if no results are given.
        deconvolution_results : pd.DataFrame, optional
            Result of :meth:`get_deconvolution`, to avoid fitting again.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``left_peak_component``,
            ``left_peak_center_time_s``, ``inflection_time_s``,
            ``inflection_heat_flow_w_g``, ``inflection_tangent_slope`` (W/g/s),
            ``x_intersection_abscissa_s`` and ``x_intersection_abscissa_min``.
        """
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
        xunit: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Determine the maximum slope of the main hydration peak.

        Runs :meth:`get_mainpeak_params` and returns only the maximum slope columns.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        show_plot : bool, optional
            Plot the analysis for each sample, by default False.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        save_path, xunit
            Deprecated, have no effect.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``gradient_from_max_slope`` (W/g/s),
            ``curvature_at_max_slope``, ``max_slope_time_s``,
            ``normalized_heat_flow_w_g_at_max_slope``,
            ``normalized_heat_j_g_at_max_slope``, the onset time
            ``onset_time_s_from_max_slope`` (and ``onset_time_min_max_slope``),
            ``normalized_heat_j_g_at_onset_time_max_slope`` and
            ``onset_time_s_max_slope_abscissa``.
        """
        _warn_ignored_args("get_maximum_slope", save_path=save_path, xunit=xunit)
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
        monochrome: bool = False,
        marker_size: float = 1.0,
    ) -> pd.DataFrame:
        """
        Characterize the main hydration peak: slopes, onset times and peak values.

        With ``method="mean"`` (default) two analyses are combined:

        * maximum slope: the maximum of the time derivative of the heat flow (after
          optional smoothing), and the onset where the tangent at this point meets
          the dormant heat flow or the time axis;
        * mean slope: a tangent averaged over the ascending flank between
          ``slope_analysis.flank_fraction_start`` and ``flank_fraction_end`` of the
          peak height, and its onset.

        With ``method="ascending"`` the slope of the first ascending segment up to
        a threshold (``slope_analysis.first_ascending_*``) is determined instead.
        This is the only analysis that honours sample parameter rules.

        In both cases the dormant period (:meth:`get_dormant_period_heatflow`) and
        ASTM C1679 (:meth:`get_astm_c1679_characteristics`) values are added.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        show_plot : bool, optional
            Plot the analysis for each sample, by default False.
        save_plot : bool, optional
            Save each plot as PNG to ``plotpath``, by default False.
        plot_type : str, optional
            For ``method="mean"``: show the "mean" (default) or "max" slope analysis.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        plotpath : pathlib.Path, optional
            Folder for ``save_plot``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, one figure per sample is created and shown.
        method : str, optional
            "mean" (default) or "ascending".
        monochrome : bool, optional
            Draw all markers in black, by default False.
        marker_size : float, optional
            Scale factor for marker sizes, by default 1.0.

        Returns
        -------
        pd.DataFrame
            One row per sample. For ``method="mean"`` the columns include
            ``gradient_from_max_slope`` (W/g/s), ``max_slope_time_s``,
            ``onset_time_s_from_max_slope``, ``onset_time_min_max_slope``,
            ``gradient_of_mean_slope`` (W/g/s), ``mean_slope_time_s``,
            ``onset_time_s_from_mean_slope``, ``peak_time_s``,
            ``normalized_heat_flow_w_g_at_peak``, ``dorm_time_s``,
            ``normalized_heat_flow_w_g_dormant`` and ``astm_time_s``, together with
            the heat flows and heats at these times. For ``method="ascending"`` the
            slope is in ``gradient_of_first_ascending_slope_to_fraction_of_max`` and
            the onset in ``onset_time_s_from_first_ascending_slope``.

        Raises
        ------
        ValueError
            If ``method`` is not "mean" or "ascending".

        Examples
        --------
        >>> processparams = ProcessingParameters()
        >>> processparams.spline_interpolation.apply = True
        >>> result = tam.get_mainpeak_params(processparams, show_plot=True, plot_type="max")
        >>> result[["sample_short", "gradient_from_max_slope", "onset_time_s_from_max_slope"]]
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
                monochrome=monochrome,
                marker_size=marker_size,
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
                monochrome=monochrome,
                marker_size=marker_size,
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
            regex=regex,
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
        monochrome: bool = False,
        marker_size: float = 1.0,
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
                    monochrome=monochrome,
                    marker_size=marker_size,
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
        monochrome: bool = False,
        marker_size: float = 1.0,
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
                monochrome=monochrome,
                marker_size=marker_size,
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
        time_discarded_s: Optional[float] = None,
        save_path: Optional[pathlib.Path] = None,
        xscale: Optional[str] = None,
        xunit: Optional[str] = None,
        intersection: Optional[str] = None,
    ):
        """
        Determine the onset of the main hydration peak from the maximum slope.

        Equivalent to ``get_mainpeak_params(processparams, plot_type="max")``.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        show_plot : bool, optional
            Plot the analysis for each sample, by default False.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        time_discarded_s, save_path, xscale, xunit, intersection
            Deprecated, have no effect. Use ``processparams.cutoff.cutoff_min`` to
            discard the initial period.

        Returns
        -------
        pd.DataFrame
            The full result of :meth:`get_mainpeak_params`. The onset is in
            ``onset_time_s_from_max_slope``.
        """
        _warn_ignored_args(
            "get_peak_onset_via_max_slope",
            time_discarded_s=time_discarded_s,
            save_path=save_path,
            xscale=xscale,
            xunit=xunit,
            intersection=intersection,
        )
        full_results = self.get_mainpeak_params(
            processparams=processparams,
            target_col=target_col,
            age_col=age_col,
            show_plot=show_plot,
            regex=regex,
            ax=ax,
            plot_type="max",
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
        flank_fraction_start: Optional[float] = None,
        flank_fraction_end: Optional[float] = None,
        window_size: Optional[float] = None,
        cutoff_min: Optional[float] = None,
        show_plot: bool = False,
        regex: Optional[str] = None,
        plotpath: Optional[pathlib.Path] = None,
        ax=None,
    ) -> pd.DataFrame:
        """
        Determine a tangent to the ascending flank of the main peak.

        Linear fits in windows along the flank between ``flank_fraction_start`` and
        ``flank_fraction_end`` of the peak height are combined to one tangent, which
        is intersected with the time axis and with the minimum heat flow before the
        peak.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        flank_fraction_start, flank_fraction_end : float, optional
            Override ``processparams.slope_analysis.flank_fraction_start`` and
            ``flank_fraction_end``.
        window_size : float, optional
            Override ``processparams.slope_analysis.window_size``.
        cutoff_min : float, optional
            Override ``processparams.cutoff.cutoff_min``.
        show_plot, plotpath, ax
            Deprecated, have no effect. Use
            ``get_mainpeak_params(show_plot=True, plot_type="mean")`` to plot.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``peak_time_s``, ``peak_value`` (W/g),
            ``tangent_slope`` (W/g/s), ``tangent_time_s``, ``tangent_value``,
            ``tangent_intercept``, ``x_intersection`` (onset on the time axis, s),
            ``x_intersection_dormant`` (onset at the minimum before the peak, s) and
            further details of the fit.
        """
        _warn_ignored_args(
            "get_ascending_flank_tangent",
            show_plot=show_plot,
            plotpath=plotpath,
            ax=ax,
        )
        params = copy.deepcopy(processparams or self.processparams)
        if flank_fraction_start is not None:
            params.slope_analysis.flank_fraction_start = flank_fraction_start
        if flank_fraction_end is not None:
            params.slope_analysis.flank_fraction_end = flank_fraction_end
        if window_size is not None:
            params.slope_analysis.window_size = window_size
        if cutoff_min is not None:
            params.cutoff.cutoff_min = cutoff_min

        analyzer = FlankTangentAnalyzer(params)
        return analyzer.get_ascending_flank_tangent(
            self._data, target_col, age_col, regex
        )

    def get_dormant_period_heatflow(
        self,
        processparams: Optional[ProcessingParameters] = None,
        regex: Optional[str] = None,
        cutoff_min: Optional[int] = None,
        upper_dormant_thresh_w_g: float = 0.002,
        plot_right_boundary: Optional[float] = None,
        prominence: Optional[float] = None,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        """
        Determine the minimum heat flow of the dormant period.

        The minimum of ``normalized_heat_flow_w_g`` between
        ``processparams.cutoff.cutoff_min`` and the first peak found by
        :meth:`get_peaks` is returned.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        cutoff_min, plot_right_boundary, prominence, show_plot
            Deprecated, have no effect. Use ``processparams.cutoff.cutoff_min`` and
            ``processparams.peakdetection.prominence``.
        upper_dormant_thresh_w_g : float, optional
            Currently has no effect.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``dorm_time_s``,
            ``normalized_heat_flow_w_g_dormant`` and ``normalized_heat_j_g_dormant``.
            Samples without a detected peak are missing.
        """
        _warn_ignored_args(
            "get_dormant_period_heatflow",
            cutoff_min=cutoff_min,
            plot_right_boundary=plot_right_boundary,
            prominence=prominence,
            show_plot=show_plot,
        )
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
        xscale: Optional[str] = None,
        xunit: str = "s",
    ) -> pd.DataFrame:
        """
        Determine the half-maximum point of the main peak according to ASTM C1679.

        For each sample, the last point on the ascending flank whose heat flow is at
        most half of the highest peak is returned.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        individual : bool, optional
            If True (default), use half of each sample's own peak height. If False,
            use half of the mean peak height of all samples.
        show_plot : bool, optional
            Plot the curves with the ASTM points, by default False.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        xscale
            Deprecated, has no effect.
        xunit : str, optional
            Time unit of the plot: "s" (default), "min", "h" or "d".

        Returns
        -------
        pd.DataFrame
            One row per sample with ``astm_time_s``,
            ``normalized_heat_flow_w_g_astm`` and ``normalized_heat_j_g_astm``.
        """
        _warn_ignored_args("get_astm_c1679_characteristics", xscale=xscale)
        params = processparams or self.processparams

        peaks = self.get_peaks(params, regex=regex, show_plot=False)

        analyzer = ASTMC1679Analyzer(params)
        df = analyzer.get_astm_c1679_characteristics(
            self._data, peaks, individual, regex
        )

        if show_plot and not df.empty:
            self._plotter.plot_astm_c1679(self._data, df, ax, xunit)

        return df

    def get_cumulative_heat_at_hours(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_h: float = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return the heat released up to a given age.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_h : float, optional
            Age in hours, by default 4.
        **kwargs
            ``cutoff_min`` is deprecated but still honoured; use
            ``processparams.cutoff.cutoff_min``.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``sample``, ``cumulative_heat_j_g`` (J/g),
            ``target_h`` and ``cutoff_min``. The deprecated column
            ``cumulated_heat_at_hours`` holds the same values and will be removed
            in a future release.

        Notes
        -----
        The heat released before ``processparams.cutoff.cutoff_min`` is subtracted.
        With the default cutoff of 60 min the result is the heat released between
        1 h and ``target_h``; set the cutoff to 0 for the total heat. For the heat
        between two arbitrary times, see :meth:`get_heat_between`.
        """
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

        return HeatCalculator.get_cumulative_heat_at_hours(
            self._data, target_h, cutoff_min
        )

    def get_cumulated_heat_at_hours(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_h: float = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Deprecated alias of :meth:`get_cumulative_heat_at_hours`.
        """
        warnings.warn(
            "get_cumulated_heat_at_hours() is deprecated and will be removed in a "
            "future release. Use get_cumulative_heat_at_hours() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_cumulative_heat_at_hours(processparams, target_h, **kwargs)

    def get_heat_between(
        self,
        start_s: Union[float, Mapping[str, float]],
        end_s: Union[float, Mapping[str, float]],
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the heat released between two times.

        The cumulative heat ``normalized_heat_j_g`` is linearly interpolated at
        ``start_s`` and ``end_s`` and the difference is returned. The cutoff of
        the processing parameters has no effect.

        Parameters
        ----------
        start_s, end_s : float or mapping
            Start and end of the interval in seconds. Either one time for all
            samples, or one time per sample as a dict or pd.Series keyed by
            ``sample_short`` (or ``sample``), e.g. an onset column of
            :meth:`get_mainpeak_params` indexed by ``sample_short``.
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``sample``, ``sample_short``, ``start_s``,
            ``end_s`` and ``heat_j_g`` (J/g). ``heat_j_g`` is NaN if a time is
            missing for the sample, lies outside the data, or ``end_s`` is before
            ``start_s``.

        Examples
        --------
        Heat released between 30 min and the onset of the main peak:

        >>> onsets = tam.get_mainpeak_params(processparams).set_index("sample_short")
        >>> tam.get_heat_between(
        ...     start_s=1800, end_s=onsets["onset_time_s_from_max_slope"]
        ... )
        """
        return HeatCalculator.get_heat_between(self._data, start_s, end_s, regex)

    def get_average_slope(
        self,
        processparams: Optional[ProcessingParameters] = None,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
        show_plot: bool = False,
        ax=None,
        save_path: Optional[pathlib.Path] = None,
        xunit: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Determine the mean slope of the ascending flank of the main hydration peak.

        Runs :meth:`get_mainpeak_params` and returns only the mean slope columns.

        Parameters
        ----------
        processparams : ProcessingParameters, optional
            Processing parameters. If None, the parameters of the Measurement are
            used.
        target_col : str, optional
            Column with the heat flow, by default "normalized_heat_flow_w_g".
        age_col : str, optional
            Column with the time in seconds, by default "time_s".
        regex : str, optional
            Regular expression; only samples whose name matches are analysed.
        show_plot : bool, optional
            Plot the analysis for each sample, by default False.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        save_path, xunit
            Deprecated, have no effect.

        Returns
        -------
        pd.DataFrame
            One row per sample with ``gradient_of_mean_slope`` (W/g/s),
            ``mean_slope_time_s``, the heat flow and heat at that time and the onset
            times and heats derived from the tangent.
        """
        _warn_ignored_args("get_average_slope", save_path=save_path, xunit=xunit)
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

    # Data manipulation methods
    def normalize_sample_to_mass(
        self, sample_short: str, mass_g: float, show_info: bool = True
    ):
        """
        Recalculate the normalized heat flow and heat of one sample for a new mass.

        Parameters
        ----------
        sample_short : str
            Name of the sample.
        mass_g : float
            Sample mass in grams.
        show_info : bool, optional
            Print a confirmation, by default True.
        """
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
            processparams = self.processparams

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
            raise ValueError(f"Failed to load metadata from {file_path}: {e}") from e

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
        downstream methods (``plot``, ``get_cumulative_heat_at_hours``, …)
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
        self._average_groupby = [groupby] if isinstance(groupby, str) else list(groupby)

    def undo_average_by_metadata(self) -> None:
        """
        Restore the individual samples after :meth:`average_by_metadata`.

        Raises
        ------
        ValueError
            If the data has not been averaged.
        """
        if not hasattr(self, "_data_before_average"):
            raise ValueError(
                "No averaged data to undo. Call average_by_metadata() first."
            )
        self._data = self._data_before_average
        del self._data_before_average
        self._average_groupby = None

    def remove_pickle_files(self):
        """
        Delete the cache files ``_data.pickle`` and ``_info.pickle``.

        The files are removed from the current working directory, if present. See
        the ``save_cache`` argument of :class:`Measurement`.
        """
        self._data_persistence.remove_pickle_files()

    # Private utility methods
    def _iter_samples(self, regex: Optional[str] = None):
        """Iterate over samples - compatibility method."""
        return SampleIterator.iter_samples(self._data, regex)
