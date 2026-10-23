# Processing Parameters

All analysis settings in CaloCem live in one object, `ProcessingParameters`.
You create it once, change the values you need and pass it to the methods of
`Measurement`:

```python
from calocem import Measurement, ProcessingParameters

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 30
processparams.spline_interpolation.apply = True

tam = Measurement(folder="calo_data", processparams=processparams)
peaks = tam.get_peaks(processparams, show_plot=False)
```

If a method is called without `processparams`, it uses the parameters the
`Measurement` was created with (or the defaults).

The parameters are grouped by topic. Each group is a small dataclass, so you
can also create a group directly, e.g.
`ProcessingParameters(cutoff=CutOffParameters(cutoff_min=30))`.

## cutoff

The beginning of a measurement is usually disturbed by mixing and thermal
equilibration. `cutoff_min` removes it from the analysis.

| Field | Default | Description |
|---|---|---|
| `cutoff_min` | `60` | Time in **minutes**. Data with `time_s < cutoff_min * 60` is ignored. `0` disables the cutoff. |

`cutoff_min` is honoured by peak detection, dormant period, maximum and mean
slope, first ascending slope, deconvolution and the related plots.

!!! note "Exceptions"
    * `get_cumulative_heat_at_hours` does not discard data; instead, the heat
      released up to `cutoff_min` is **subtracted** from the heat at `target_h`.
    * `get_peak_onsets` ignores `cutoff_min` and uses its own
      `time_discarded_s` argument.
    * The half-maximum search of `get_astm_c1679_characteristics` uses the
      unfiltered data; only the peak detection it relies on honours the cutoff.

## time_constants

Time constants of the calorimeter for the [Tian correction](tian.md), in
seconds. Used by `apply_tian_correction` only.

| Field | Default | Description |
|---|---|---|
| `tau1` | `300` | First time constant. |
| `tau2` | `100` | Second time constant. Set to `None` to correct with `tau1` only. |

## peakdetection

Passed to [`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)
when searching peaks in the heat flow.

| Field | Default | Description |
|---|---|---|
| `prominence` | `1e-5` | Minimum peak prominence, in the unit of the analysed column (W/g). Decrease it to find smaller peaks. |
| `distance` | `100` | Minimum distance between peaks as a **number of data points**, not a time. |

Peak detection is the first step of `get_peaks`, `get_dormant_period_heatflow`,
`get_astm_c1679_characteristics` and `get_mainpeak_params`.

## Smoothing

Derivatives of measured heat flow are noisy. Three groups control smoothing.
They act at different points, so it matters which analysis you run:

| Group | Fields (default) | What is smoothed |
|---|---|---|
| `rolling_mean` | `apply` (`False`), `window` (`11` points) | Heat flow, before the gradient is taken. Used by the maximum slope analysis and `plot_heatflow_with_gradient`. |
| `median_filter` | `apply` (`False`), `size` (`7` points) | The first and second **derivative** of the heat flow (not the heat flow itself). |
| `spline_interpolation` | `apply` (`False`), `smoothing_1st_deriv` (`1e-9`), `smoothing_2nd_deriv` (`1e-9`) | A cubic smoothing spline is fitted to the first and second derivative. The values are passed as `s` to [`UnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html); larger values mean smoother curves. |

The smoothed derivatives are used by the maximum slope analysis
(`get_maximum_slope`, `get_mainpeak_params`, `get_peak_onset_via_max_slope`),
`plot_heatflow_with_gradient` and `apply_tian_correction`. The mean slope,
first ascending slope, peak detection and deconvolution work on the
unsmoothed data.

## downsample

Large files can be thinned out when the `Measurement` is created. Points are
kept preferentially where the heat flow curve bends, so peaks stay sharp
while flat regions are thinned out. Values are picked from the measured
data, not interpolated.

| Field | Default | Description |
|---|---|---|
| `apply` | `False` | Enable downsampling. |
| `num_points` | `1000` | Target number of points per sample (an upper bound). |
| `smoothing_factor` | `1e-6` | Smoothing of the spline used to estimate the curvature. |
| `baseline_weight` | `0.1` | Share of points spread evenly, so flat regions are not emptied. |
| `section_split` | `False` | Downsample the data before and after `section_split_time_s` separately. |
| `section_split_time_s` | `1000` | Split time in seconds. |

Downsampling only takes effect when passed to the constructor:
`Measurement(folder, processparams=processparams)`. Save the result with
`tam.save(...)` to avoid repeating it, see [Loading data](loading.md#saving-and-reloading).

## slope_analysis

Controls the analyses of the main hydration peak, see
[Main Peak Parameters](example_get_mainpeak_params.ipynb).

| Field | Default | Description |
|---|---|---|
| `flank_fraction_start` | `0.35` | Start of the ascending flank, as fraction of the height between the minimum before the peak and the peak. |
| `flank_fraction_end` | `0.55` | End of the ascending flank, same scale. |
| `window_size` | `0.1` | Width of the averaging window for the mean slope, as fraction of the time span of the flank. |
| `first_ascending_fraction_of_max` | `0.2` | `method="ascending"`: heat flow threshold as fraction of the maximum (after cutoff). Must be in (0, 1]. |
| `first_ascending_range_method` | `"fraction"` | `method="ascending"`: `"fraction"` uses the flank fractions above, `"delta"` uses a fixed heat-flow rise. |
| `first_ascending_delta_y_w_g` | `0.001` | `"delta"` mode: rise of the heat flow in W/g that defines the segment. |
| `flexible` | `0.0` | `"delta"` mode: widens the delta for high peaks; `0` disables it. |

## deconvolution

Used by `get_deconvolution` and `get_left_peak_inflection_tangent_intersection`,
which fit a sum of peaks plus a baseline to the heat flow.

| Field | Default | Description |
|---|---|---|
| `n_peaks` | `2` | Number of peaks to fit. |
| `baseline_mode` | `"constant"` | `"constant"`, `"linear"`, `"chebyshev"` or `"none"`. |
| `chebyshev_degree` | `2` | Polynomial degree for `baseline_mode="chebyshev"`. |
| `max_nfev` | `5000` | Maximum number of optimizer iterations. |
| `max_fit_points` | `2500` | The data is subsampled to at most this many points for the fit. |
| `min_points` | `30` | Samples with fewer points (after cutoff) are skipped. |
| `min_peak_distance_fraction` | `0.05` | Minimum distance of the initial peak guesses, as fraction of the number of points. |
| `min_sigma_fraction`, `max_sigma_fraction` | `0.005`, `0.3` | Width bounds of Gaussian peaks, as fraction of the time range. |
| `min_peak_time_separation_fraction` | `1e-4` | Minimum separation of the fitted peak times, as fraction of the time range. |
| `relative_intensity_upper_bounds` | `None` | Optional list (one value per peak, each in (0, 1]) capping each peak's share of the total amplitude. |
| `peak_width_upper_bounds` | `None` | Optional list (one value per peak) of maximum peak widths. |

The peak shape is selected with the `peak_shape` argument of
`get_deconvolution` (`"lognormal"` or `"gaussian"`); the `peak_shape` field is
currently ignored.

## plotting

Used by the analysis plots of `get_mainpeak_params` (and the methods built on
it) and, for `figsize`, by `plot_heatflow_with_gradient`. `Measurement.plot`
does not use these settings.

| Field | Default | Description |
|---|---|---|
| `figsize` | `(10, 6)` | Figure size in inches, used when no `ax` is passed. |
| `time_unit` | `"seconds"` | `"seconds"`, `"minutes"` or `"hours"`. |
| `heat_unit` | `"W"` | `"W"` or `"mW"`. |
| `show_plot_title` | `True` | Show a title above each plot. |
| `plot_title` | `"filename"` | With metadata loaded: a list of metadata column names whose values form the title, e.g. `["cement_name", "w_c"]`. |
| `legend_pos` | `"best"` | `"best"` or `"outside"` (right of the plot). |
| `xlims`, `ylims` | `None` | Axis limits as `(min, max)` in plot units. `ylims` is only applied together with `xlims`. |

## Sample-specific parameters

Different samples sometimes need different settings. Rules map a regular
expression on the sample name to parameter overrides:

```python
processparams.add_sample_param_rule(
    r"^c3a",  # samples whose name starts with "c3a"
    {"slope_analysis": {"first_ascending_range_method": "delta"}},
)
```

Rules with a higher `priority` are applied later and therefore win.

!!! warning
    Sample rules are currently only honoured by
    `get_mainpeak_params(method="ascending")`. All other analyses use the base
    parameters for every sample.

## Reserved parameters

The groups `gradient_peakdetection`, `nonlin_savgol` and `preprocess` and the
field `cutoff.cutoff_max` exist but are not used by any analysis yet.
Changing them has no effect.
