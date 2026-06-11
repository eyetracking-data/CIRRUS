# CIRRUS Evaluation

This repository contains a preliminary evaluation of CIRRUS, an interactive framework for designing preprocessing pipelines for eye-tracking data.

## Evaluation Goals

The evaluation addresses three questions:

* **RQ1:** Does CIRRUS produce different preprocessing recommendations for datasets with different quality profiles?
* **RQ2:** Do the rule-based recommendations react systematically to controlled data degradations?
* **RQ3:** Are CIRRUS-recommended pipelines competitive in downstream classification?

## Datasets

The evaluation uses two eye-tracking datasets:

1. **Autism eye-tracking dataset**
   Used as a comparatively clean, low-noise dataset.

2. **Academic reading / cheating dataset**
   Used as a noisier behavioral dataset with substantial missingness and stronger variability. For the classification experiment, labels are derived from the directory structure:

   * `Schummeln` → cheating = 1
   * `Nicht_Schummeln` → cheating = 0

## Feature Profiling

For each numerical feature, CIRRUS computes:

* missing-value rate,
* IQR-based outlier rate,
* skewness,
* kurtosis.

These indicators are then mapped to preprocessing recommendations for:

* missing-value imputation,
* outlier handling,
* normalization.

In the evaluated data, the autism dataset showed low corruption, with an average missing-value rate of 0.38%. The academic reading dataset was substantially noisier, with an average missing-value rate of 32.43%. CIRRUS therefore recommended simpler preprocessing pipelines for the autism data and more robust preprocessing pipelines for the academic reading data.

## Controlled Degradation Test

To test whether the heuristic rules behave systematically, selected signals were artificially degraded by injecting:

* missing values,
* outliers,
* skewness.

The recommendation logic changed consistently with the induced data quality problems:

* Increasing missingness shifted imputation recommendations from LOCF/mean imputation toward KNN/MICE.
* Increasing outlier contamination shifted outlier handling toward Isolation Forest.
* Strong skewness led to MAD/Isolation Forest and RobustScaler recommendations.

This experiment supports the interpretation that CIRRUS does not apply arbitrary defaults, but uses explicit and reproducible rule-based mappings.

## Downstream Classification

For the academic reading / cheating dataset, we compared four preprocessing variants:

| Pipeline           | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| Raw aggregate      | Aggregated features without preprocessing                  |
| Simple default     | Mean imputation + simple scaling                           |
| Robust default     | Robust preprocessing baseline                              |
| CIRRUS recommended | Feature-wise preprocessing based on CIRRUS recommendations |

Using Random Forest classification with five-fold cross-validation, the CIRRUS-recommended pipeline achieved the best accuracy and F1 score among the tested variants and remained competitive in AUC.

The evaluation should be interpreted as preliminary. CIRRUS is not claimed to identify the globally optimal preprocessing pipeline. Instead, it provides transparent, reproducible, and data-dependent preprocessing recommendations that can be inspected, adjusted, and compared against alternative baselines.

## Generated Files

The evaluation produces the following result files:

* `cirrus_feature_profile_recommendations.csv`
* `cirrus_recommendation_summary.csv`
* `cirrus_controlled_degradation_test.csv`
* `cirrus_imputation_benchmark.csv`
* `cirrus_downstream_ml_comparison.csv`

These files document the profiling results, recommendation behavior, controlled degradation tests, and downstream classification comparison.



----

### Evaluation

To address the heuristic nature of the recommendation logic, we conducted a preliminary evaluation of CIRRUS on two heterogeneous eye-tracking datasets. The evaluation follows three goals: (RQ1) to test whether CIRRUS produces data-dependent recommendations for different quality profiles, (RQ2) to analyze whether the rule-based recommendations react systematically to controlled degradations, and (RQ3) to assess whether CIRRUS-recommended preprocessing remains competitive in downstream classification.

First, we profiled all numerical features of the autism and academic reading datasets using the same indicators as in CIRRUS: missing-value rate, IQR-based outlier rate, skewness, and kurtosis. The autism dataset showed low corruption, with an average missing-value rate of 0.38%, whereas the academic reading dataset was substantially noisier, with an average missing-value rate of 32.43%. Accordingly, CIRRUS recommended simpler preprocessing pipelines for the low-noise dataset and more robust pipelines for the noisier behavioral dataset.

Second, we performed controlled degradation experiments by injecting artificial missing values, outliers, and skewness into selected signals. The recommendation logic reacted consistently to these changes: increasing missingness shifted the recommended imputation strategy from LOCF or mean imputation toward KNN or MICE; increasing outlier contamination shifted the outlier strategy from Z-score/IQR-based methods toward Isolation Forest; and strong skewness led to robust outlier handling and robust scaling. This shows that the heuristic rules are not arbitrary defaults, but explicit and reproducible mappings from measurable data-quality indicators to preprocessing strategies.

Third, we evaluated downstream classification on the academic reading dataset, where class labels were derived from the directory structure. We compared raw feature aggregation, a simple default pipeline, a robust default pipeline, and the CIRRUS-recommended pipeline using Random Forest classification with five-fold cross-validation. CIRRUS achieved the best accuracy and F1 score among the tested variants, while remaining competitive in AUC. These results do not imply that CIRRUS always identifies the globally optimal preprocessing pipeline, but they indicate that its recommendations provide a transparent and empirically useful alternative to arbitrary preprocessing defaults.

The full evaluation notebook, generated result tables, and reproducibility material are provided in the accompanying GitHub repository.

