# KNN vs NC Research Framework

**A Critique of:**
> *Comparative Analysis Between the Traditional and Nearest Centroid Methods in the K-Nearest Neighbor Algorithm*
> Mladenova & Valova, IEEE HORA 2022

**Hypothesis:** The benchmark datasets used in the original paper exhibit distributional properties (central tendency, unimodality, low skewness) that inherently favour the Nearest Centroid classifier. The reported similarity between KNN and NC is an artefact of dataset bias — not evidence of general equivalence.

---

## Project Structure

```
knn_nc_research/
│
├── datasets/                        One class per UCI dataset
│   ├── base_dataset.py              Fetch + cache + clean (abstract)
│   ├── iris_dataset.py
│   ├── wine_dataset.py
│   ├── ionosphere_dataset.py
│   ├── ecoli_dataset.py
│   ├── glass_dataset.py
│   ├── hepatitis_dataset.py
│   ├── lenses_dataset.py
│   ├── lung_cancer_dataset.py
│   ├── soybean_dataset.py
│   └── thyroid_dataset.py
│
├── statistical_tests/               One class per test
│   ├── base_test.py                 Abstract base with run_test() interface
│   ├── normality_test.py            Henze-Zirkler multivariate normality
│   ├── unimodality_test.py          Hartigan Dip Test per feature
│   ├── skewness_test.py             Skewness per feature
│   ├── kurtosis_test.py             Excess kurtosis per feature
│   └── separability_test.py         Centroid distance / intra-class spread
│
├── visualization/                   One class per plot type
│   ├── base_plot.py                 Abstract base with draw() interface
│   ├── histogram.py                 Histogram + KDE + Gaussian reference
│   ├── qq_plot.py                   Q-Q plots per class per feature
│   ├── pca_scatter.py               PCA 2D with centroid markers
│   ├── boxplot.py                   Box plots per feature by class
│   ├── density_plot.py              Smooth KDE density curves
│   └── summary_heatmap.py           Cross-dataset heatmap + bar chart
│
├── synthesis/                       One class per distribution type
│   ├── base_generator.py            Abstract base with generate() interface
│   ├── central_tendency_generator.py  Tight Gaussian — NC ideal
│   ├── normal_generator.py          Standard Gaussian blobs
│   ├── uniform_generator.py         Uniform distributions
│   ├── skewed_generator.py          Log-normal / skewed
│   ├── heavy_tail_generator.py      Student's t — heavy tails
│   ├── multimodal_generator.py      Multiple clusters per class
│   ├── imbalanced_generator.py      Majority/minority class ratio
│   ├── non_central_generator.py     Scattered, overlapping — NC worst case
│   └── mixed_generator.py           Mixed Gaussian + log-normal + uniform
│
├── experiments/
│   ├── knn_vs_nc.py                 KNNvsNCExperiment — CV accuracy, F1, time
│   └── distribution_sweep.py        DistributionSweepExperiment — all scenarios
│
├── data/                            Auto-created; cached .parquet files
├── main.py                          Demonstration script
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## API — Everything by Example

### Load a Dataset (with automatic caching)

```python
from datasets import IrisDataset

iris = IrisDataset()
df   = iris.load()               # downloads once, caches to data/iris.parquet

print(iris.summary())
features = iris.features()
subset   = iris.class_subset("Iris-setosa")
```

### Run a Statistical Test

```python
from statistical_tests import NormalityTest, SeparabilityTest

# Per-class test
normality = NormalityTest(subset, alpha=0.05)
result    = normality.run_test()
print(result["is_normal"])
print(result["interpretation"])

# Full-dataset separability
sep    = SeparabilityTest(df, target_col="target")
result = sep.run_test()
print(result["avg_ratio"])
```

Every test uses the same interface: `test = SomeTest(df, ...)` → `result = test.run_test()`

### Draw a Visualisation

```python
from visualization import Histogram, QQPlot, PCAScatter, BoxPlot, DensityPlot

Histogram(df, target_col="target", title="Iris Histogram", bins=12).draw()
QQPlot(df, target_col="target").draw()
PCAScatter(df, target_col="target").draw()
BoxPlot(df, target_col="target").draw()
DensityPlot(df, target_col="target").draw()

# Save to file
PCAScatter(df, title="Iris PCA").draw(save_path="reports/iris_pca.png")
```

Every plot uses the same interface: `plot = SomePlot(df, ...)` → `plot.draw()`

### Generate Synthetic Data

```python
from synthesis import (
    CentralTendencyGenerator, NormalGenerator, MultimodalGenerator,
    SkewedGenerator, HeavyTailGenerator, NonCentralGenerator,
)

df_normal    = NormalGenerator(n_samples=600, n_features=4, separation=4.0).generate()
df_multimodal = MultimodalGenerator(n_samples=600, modes_per_class=2).generate()
df_skewed    = SkewedGenerator(n_samples=600, skewness=1.5).generate()
df_nc_ideal  = CentralTendencyGenerator(n_samples=600, spread=0.3).generate()
df_nc_worst  = NonCentralGenerator(n_samples=600, spread=3.5, separation=1.5).generate()
```

Every generator uses the same interface: `gen = SomeGenerator(...)` → `df = gen.generate()`

### Run an Experiment

```python
from experiments import KNNvsNCExperiment, DistributionSweepExperiment

# Single dataset experiment
exp    = KNNvsNCExperiment(df, k=5, cv_folds=10)
result = exp.run()
print(result["comparison_df"])        # accuracy, F1, time — KNN vs NC
print(result["knn_summary"])
print(result["nc_summary"])

# Full sweep across all synthetic distribution types
sweep   = DistributionSweepExperiment(n_samples=600, k=5, cv_folds=10)
results = sweep.run()
print(results["comparison"])          # one row per scenario
print(results["winner_counts"])       # {"KNN": N, "NC": N, "Tie": N}
```

---

## Adding New Components

### New Dataset

```python
# datasets/my_dataset.py
from datasets.uci_dataset_loader import BaseDataset


class MyDataset(BaseDataset):
    def _dataset_name(self): return "my_dataset"

    def _uci_id(self):       return 999
```

Register it in `datasets/__init__.py`. Done — caching, cleaning, and all methods work automatically.

### New Statistical Test

```python
# statistical_tests/my_test.py
from statistical_tests.base_test import BaseTest

class MyTest(BaseTest):
    def run_test(self):
        # ... compute stats on self._df
        return {"test_name": "My Test", "result": ..., "interpretation": "..."}
```

### New Visualization

```python
# visualization/my_plot.py
from visualization.base_plot import BasePlot

class MyPlot(BasePlot):
    def _build_figure(self):
        fig, ax = plt.subplots()
        # ... draw on ax using self._df
        return fig
```

### New Generator

```python
# synthesis/my_generator.py
from synthesis.base_generator import BaseGenerator

class MyGenerator(BaseGenerator):
    def _generate_data(self):
        # return X (np.ndarray), y (np.ndarray)
        return X, y
```

---

## Experiment Metrics

`KNNvsNCExperiment.run()` returns:

| Key | Type | Description |
|---|---|---|
| `comparison_df` | DataFrame | Side-by-side metric table |
| `knn_summary` | dict | Mean ± std across folds for KNN |
| `nc_summary` | dict | Mean ± std across folds for NC |
| `knn_confusion` | ndarray | Aggregated confusion matrix |
| `nc_confusion` | ndarray | Aggregated confusion matrix |
| `dataset_info` | dict | n_instances, n_features, n_classes, k, folds |

Metrics computed: accuracy, precision (weighted), recall (weighted), F1 (weighted), execution time (ms).
