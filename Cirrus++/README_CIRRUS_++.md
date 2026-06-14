# CIRRUS++ – Domain-Aware Eye-Tracking Preprocessing Assistant

This repository contains two closely related versions of the CIRRUS prototype:

- **CIRRUS 1.0**: the baseline demo-paper version. It implements the original visual preprocessing assistant with feature profiling, rule-based recommendations, method previews, and export.
- **CIRRUS++**: an extended research prototype that keeps the same general CIRRUS idea and preprocessing engine, but adds stronger profiling, domain-aware recommendation scoring, pipeline-order analysis, composite quality scoring, synthetic ground-truth validation, and optional downstream validation.

The demo paper should still be read as describing **CIRRUS 1.0**. The file `cirrus++.py` is the improved prototype version used to explore the reviewer feedback and possible future extensions.

---

## 1. What CIRRUS does

Eye-tracking data are noisy time-series data. Typical problems are missing gaze samples, blink-related gaps, off-screen points, pupil artifacts, calibration drift, skewed distributions, and unstable derived signals such as gaze velocity. CIRRUS supports users in designing preprocessing pipelines instead of choosing preprocessing steps blindly.

The main workflow is:

```text
Upload CSV
→ select numerical eye-tracking features
→ profile data quality
→ receive preprocessing recommendations
→ compare methods visually and quantitatively
→ execute pipeline
→ export processed data and pipeline report
```

CIRRUS 1.0 focuses on the core workflow:

```text
missing values → outlier handling → normalization → optional smoothing/filtering
```

CIRRUS++ extends this into a more explicit audit and evaluation environment.

---

## 2. Running CIRRUS++

Install the required Python packages:

```bash
pip install streamlit pandas numpy scipy scikit-learn plotly
```

Run the app:

```bash
streamlit run cirrus++.py
```

Then open the Streamlit URL shown in the terminal.

---

## 3. Input data format

The app expects a CSV file with numerical time-series columns. Typical eye-tracking columns are for example:

```text
Gaze_X
Gaze_Y
Pupil_Left
Pupil_Right
Gaze_Velocity
Gaze_Acceleration
Fixation_Duration_ms
Saccade_Duration_ms
```

Metadata columns may exist, but should usually **not** be selected as preprocessing features:

```text
Sample_ID
Participant
Task
Stimulus
Condition
Condition_Code
Sampling_Rate_Hz
Row
Time_s
```

These metadata columns are useful for alignment, grouping, labels, or validation, but they should not be normalized or smoothed like gaze or pupil signals.

---

## 4. Main features of CIRRUS++

### 4.1 CSV loading and automatic numeric conversion

CIRRUS++ can load CSV files with configurable delimiters and decimal formats. It tries to convert quasi-numeric columns automatically, including values with decimal commas or common missing-value tokens.

Useful sidebar settings:

- CSV delimiter: `Auto`, comma, semicolon, tab, pipe
- Decimal format: dot or comma
- Include likely ID/time columns as features: normally leave this off

---

### 4.2 Domain-aware profiling

At the beginning, the user chooses or auto-detects a domain profile. The current profiles are:

- Auto-detect
- Clinical / diagnostic pupil data
- Academic reading / cognitive task
- High-frequency lab tracking
- Web/mobile remote tracking
- Multimodal behavioral / sensor fusion

The selected domain changes the weighting of quality metrics and shifts the method priors used by the recommender. This does not mean that the app magically knows the true domain; it is a transparent configuration layer that makes assumptions explicit.

---

### 4.3 Quality dashboard

For selected features, CIRRUS++ computes a feature-level quality profile. The dashboard includes:

- Missing-value rate
- Longest missing run
- Missing burstiness
- IQR outlier rate
- MAD outlier rate
- Skewness
- Excess kurtosis
- Zero-variance flag
- Unique-value ratio
- Drift score
- Volatility score
- Entropy score
- Mean absolute feature correlation
- Composite Quality Index

The **Quality Index** is a 0–100 score. It combines several quality dimensions:

```text
completeness
temporal continuity
plausibility / outlier load
distributional shape
stability / drift
multivariate correlation
```

Interpretation:

```text
80–100  good
60–79   usable, inspect
40–59   risky
0–39    critical
```

The Quality Index is not a ground-truth measure. It is an intrinsic screening score that helps compare features and pipelines.

---

### 4.4 Adaptive recommendation scoring

CIRRUS 1.0 used a simpler heuristic logic based mainly on missingness, outliers, and skewness. CIRRUS++ keeps that idea but scores candidate methods from more signals:

```text
missingness level
missingness pattern / burstiness
outlier load
skewness
kurtosis
drift
volatility
feature correlations
domain profile
recommendation aggressiveness
```

The recommendation contains one method from each stage:

```text
Missing-value handling
Outlier handling
Normalization
Smoothing / filtering
```

Available methods include:

**Missing values**

- None
- Mean Imputation
- LOCF
- Linear Interpolation
- KNN Imputation
- MICE

**Outliers**

- None
- Z-Score
- IQR
- MAD
- Winsorization
- Isolation Forest

**Normalization**

- None
- StandardScaler
- RobustScaler
- MinMaxScaler

**Smoothing / filtering**

- None
- Butterworth
- Savitzky-Golay
- Fourier smoothing

For larger time-series tables, CIRRUS++ uses scalable temporal fallbacks for expensive imputers such as KNN/MICE. Exact KNN imputation can become memory-intensive on long eye-tracking streams.

---

## 5. Important interface sections

### 5.1 Feature-level recommendation

This section shows one selected feature in detail. It reports the Quality Index, missingness, outlier level, skewness/kurtosis, and the recommended preprocessing pipeline.

Use this section to answer:

```text
Why does CIRRUS recommend this pipeline for this feature?
```

---

### 5.2 Signal preview

This section shows the raw signal and the processed signal after the selected CIRRUS++ pipeline.

Use this section to answer:

```text
What does the final pipeline do to the signal?
```

---

### 5.3 Order-aware step-wise pipeline audit

This is one of the most important additions in CIRRUS++.

It shows the pipeline **stage by stage**, for example:

```text
Raw
→ 1 Missing: Linear Interpolation
→ 2 Outlier: MAD
→ 2b Re-fill after outlier
→ 3 Normalization: RobustScaler
→ 4 Smoothing: Savitzky-Golay
```

Alternatively, the user can reverse the order:

```text
Raw
→ 1 Outlier: MAD
→ 2 Missing: Linear Interpolation
→ 3 Normalization: RobustScaler
→ 4 Smoothing: Savitzky-Golay
```

This is controlled in the sidebar:

```text
Pipeline order:
Missing → Outlier
Outlier → Missing
```

This matters because missing-value imputation and outlier handling interact. If outliers are removed after imputation, new gaps can appear. If outliers are removed before imputation, the imputation method reconstructs both original gaps and artifact gaps.

The audit includes:

- stage-wise overlay plot
- stage-wise metric table
- Quality Index per stage
- mean absolute change from raw
- mean absolute change from previous stage
- missingness and imputation impact
- outlier rates
- drift, volatility, entropy, jitter proxy
- spatial and sequential metrics for gaze coordinates

This is the best section for demonstrating that CIRRUS++ makes preprocessing inspectable rather than implicit.

---

### 5.4 Method comparison lab

This section compares method families independently:

- Missing-value methods
- Outlier methods
- Normalization methods
- Smoothing methods
- Pipeline grid combinations

The method comparison lab is useful for seeing how methods differ within one stage. It is less useful for understanding pipeline-order effects. For order effects, use the **Order-aware step-wise pipeline audit**.

---

### 5.5 MV × Outlier interaction grid

This section compares combinations of missing-value methods and outlier methods. It helps answer:

```text
Which imputation method works well together with which outlier strategy?
```

The grid ranks combinations by intrinsic metrics such as Quality Index, missingness, outlier level, skewness, and kurtosis.

---

### 5.6 Optional downstream validation

If the dataset contains a target column, CIRRUS++ can run a lightweight baseline-vs-processed validation.

Example target columns:

```text
Condition
Condition_Code
Class
Label
Diagnosis
TaskType
```

The app compares:

```text
raw features → model → target
processed features → model → target
```

This is useful when the dataset has labels and the goal is predictive modeling.

Do **not** use a signal column such as `Gaze_X` as target. `Gaze_X` is an input feature, not the thing to be predicted.

For the synthetic datasets in this repository, use:

```text
Target column: Condition_Code
```

or:

```text
Target column: Condition
```

`Condition_Code` is usually simpler because it is already numeric.

---

### 5.7 Ground-truth validation

This is the strongest validation mode for the synthetic dataset.

Workflow:

1. Upload the noisy CSV as the main dataset.
2. Scroll to **Ground-truth validation**.
3. Upload the matching clean CSV as ground-truth file.
4. Select an alignment column, preferably `Sample_ID`.
5. Select features such as `Gaze_X`, `Gaze_Y`, `Pupil_Left`, `Pupil_Right`.
6. Compare raw noisy signal vs. CIRRUS++ processed signal against the clean ground truth.

The app computes metrics such as:

```text
Raw MAE
CIRRUS MAE
MAE gain %
Raw RMSE
CIRRUS RMSE
RMSE gain %
Raw correlation
CIRRUS correlation
```

Positive gain means:

```text
CIRRUS++ is closer to the clean ground truth than the noisy raw signal.
```

For MAE/RMSE, prefer the reconstruction stage before normalization. Once a signal is normalized, the scale no longer matches the clean reference unless both are transformed identically.

---

## 6. Synthetic datasets

This repository includes two small synthetic datasets for testing CIRRUS++:

```text
cirrus_synthetic_ground_truth_clean_small.csv
cirrus_synthetic_noisy_injected_small.csv
```

Both files contain the same rows and can be aligned by:

```text
Sample_ID
```

Dataset size:

```text
48,600 rows
18 columns
balanced target: cheating / non_cheating
```

Important columns:

```text
Sample_ID
Participant
Task
Stimulus
Condition
Condition_Code
Sampling_Rate_Hz
Row
Time_s
Gaze_X
Gaze_Y
Pupil_Left
Pupil_Right
Gaze_Velocity
Gaze_Acceleration
Fixation_Duration_ms
Saccade_Duration_ms
Blink_Flag
```

The clean file contains the synthetic ground-truth time series.

The noisy file contains injected disturbances such as:

```text
Gaussian sensor jitter
calibration drift
blink-like clustered missingness
longer tracking loss
off-screen gaze spikes
pupil artifacts
```

In the noisy file, the main gaze and pupil channels contain approximately 10% missing values:

```text
Gaze_X
Gaze_Y
Pupil_Left
Pupil_Right
```

---

## 7. Recommended demo workflow with the synthetic data

### 7.1 Basic preprocessing demo

1. Run the app:

```bash
streamlit run cirrus++.py
```

2. Upload:

```text
cirrus_synthetic_noisy_injected_small.csv
```

3. Use these CSV settings if auto-detection fails:

```text
Delimiter: comma
Decimal: dot
```

4. Select signal features:

```text
Gaze_X
Gaze_Y
Pupil_Left
Pupil_Right
Gaze_Velocity
Gaze_Acceleration
Fixation_Duration_ms
Saccade_Duration_ms
```

5. Do not select metadata columns as features:

```text
Sample_ID
Participant
Task
Stimulus
Condition
Condition_Code
Sampling_Rate_Hz
Row
Time_s
```

6. Inspect:

```text
Quality dashboard
Feature-level recommendation
Signal preview
Order-aware step-wise pipeline audit
Method comparison lab
MV × Outlier interaction grid
```

---

### 7.2 Pipeline-order demo

Use the sidebar setting:

```text
Pipeline order
```

Compare:

```text
Missing → Outlier
```

against:

```text
Outlier → Missing
```

Then inspect the **Order-aware step-wise pipeline audit**. This should show how the intermediate signal and stage-wise metrics change.

This is useful because it demonstrates that preprocessing is not just a list of methods. The order of operations is itself a modeling decision.

---

### 7.3 Ground-truth validation demo

1. Upload the noisy file as the main CSV:

```text
cirrus_synthetic_noisy_injected_small.csv
```

2. In the **Ground-truth validation** section, upload:

```text
cirrus_synthetic_ground_truth_clean_small.csv
```

3. Use:

```text
Alignment column: Sample_ID
```

4. Select ground-truth features:

```text
Gaze_X
Gaze_Y
Pupil_Left
Pupil_Right
```

5. Interpret:

```text
Positive MAE gain or RMSE gain = CIRRUS++ reconstruction is closer to clean ground truth.
Higher CIRRUS correlation = processed signal follows the clean signal better.
```

---

### 7.4 Downstream validation demo

1. Upload the noisy file.
2. Select signal features.
3. In **Optional downstream validation**, choose:

```text
Condition_Code
```

4. The app compares a lightweight model on raw features against the same model on processed features.

This is not a replacement for a full experimental evaluation. It is a quick sanity check that preprocessing can affect predictive performance.

---

## 8. Exported files from the app

CIRRUS++ can export:

```text
cirrus_processed.csv
cirrus_pipeline.json
cirrus_quality_report.json
```

Use these files for reproducibility:

- `cirrus_processed.csv`: cleaned/processed data
- `cirrus_pipeline.json`: selected methods and parameter settings
- `cirrus_quality_report.json`: before/after quality profiles and comparison tables

---

## 9. Relation to the demo paper

The demo paper describes the original CIRRUS concept: an interactive visual framework that profiles eye-tracking data, recommends preprocessing methods, previews effects, and exports cleaned data.

CIRRUS++ should be described in the repository as an extended prototype, not as the exact system described in the demo paper.

Suggested wording:

```text
The demo paper describes CIRRUS 1.0, the baseline visual preprocessing assistant.
This repository additionally contains CIRRUS++, an extended prototype that builds on
the same idea and preprocessing engine but adds domain-aware profiling, composite
quality scoring, pipeline-order auditing, method interaction grids, synthetic
ground-truth validation, and optional downstream validation.
```

This distinction is important because CIRRUS++ contains functionality beyond the submitted demo paper.

---

## 10. Current limitations

CIRRUS++ is a research prototype.

Known limitations:

- It runs locally in Streamlit and is not yet a production-scale backend.
- Exact KNN/MICE imputation can be slow or memory-intensive for very large datasets.
- Some metrics are intrinsic quality indicators, not proof of scientific correctness.
- Domain profiles encode assumptions and should be inspected critically.
- Ground-truth validation is only possible when a clean aligned reference file exists.
- Downstream validation is only meaningful when a real target label exists.
- For participant-level studies, proper group-aware validation should be added before claiming final predictive performance.

For a stronger data-management version, the next step would be a chunked backend using DuckDB, Polars, Arrow, or Dask, plus group-aware cross-validation.

---

## 11. Minimal repository structure

A clean GitHub structure could look like this:

```text
.
├── cirrus_1_0.py
├── cirrus++.py
├── data/
│   ├── cirrus_synthetic_ground_truth_clean_small.csv
│   └── cirrus_synthetic_noisy_injected_small.csv
├── README.md
└── requirements.txt
```

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scipy
scikit-learn
plotly
```

---

## 12. Quick start

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run cirrus++.py
```

Then upload:

```text
data/cirrus_synthetic_noisy_injected_small.csv
```

For ground-truth validation, additionally upload:

```text
data/cirrus_synthetic_ground_truth_clean_small.csv
```
