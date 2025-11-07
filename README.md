# Analysis Utilities

## Parameter Delta Analysis

Use `visualization/parameter_delta_analysis.py` to compare the parameters of an
original model checkpoint against its defended counterpart and visualize the
layer-wise differences.

```bash
python visualization/parameter_delta_analysis.py \
    path/to/original_model.pt \
    path/to/defended_model.pt \
    --output-dir results/visualization/parameter_deltas \
    --metric l2 \
    --agg sum
```

The script writes aggregated statistics to `parameter_delta_summary.csv` and
saves both heatmap and line-chart visualizations in the specified output
directory. Select from `l2`, `mean_abs`, or `max_abs` metrics and aggregate them
with `sum`, `mean`, or `max` to tailor the analysis.
