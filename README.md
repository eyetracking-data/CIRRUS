CIRRUS – Clean Intelligent Refinement for Real-time User Sight Data
An Interactive Visual Framework for Preprocessing Eye-Tracking Data

CIRRUS is a Streamlit-based application that provides an interactive and transparent workflow for preprocessing eye-tracking signals. It implements empirically grounded heuristics for missing-value imputation, outlier detection, and normalization, and visualizes the effect of different preprocessing strategies in real time. The system is designed for researchers who want to replace trial-and-error preprocessing with a structured and explainable process.

---------------------------------------
FEATURES
---------------------------------------
- Automated profiling of eye-tracking features (missingness, outlier rate, skewness, kurtosis)
- Rule-based recommendations for imputation, outlier handling, and normalization
- Interactive visualization of multiple preprocessing methods for direct comparison
- Configurable pipeline execution (missing values → outliers → scaling)
- Summary statistics before and after processing
- CSV export of the cleaned data
- Optional visual smoothing (Butterworth filter, Fourier smoothing, clipping)

---------------------------------------
INSTALLATION
---------------------------------------
1. Clone the repository:
   git clone <your-repo-url>
   cd CIRRUS

2. Install dependencies (Python 3.11 recommended):
   
Required packages include:
streamlit
pandas
numpy
scikit-learn
scipy
plotly

---------------------------------------
RUNNING THE APPLICATION
---------------------------------------
Start the app locally:

   streamlit run eye-tracking-app2.py

The application will open in your browser at:
   http://localhost:8501

---------------------------------------
USAGE OVERVIEW
---------------------------------------
1. Upload a CSV file.
2. Inspect data quality (missing rate, outlier rate, skewness, kurtosis).
3. View heuristic recommendations for:
   - missing value imputation (LOCF, KNN, MICE, etc.)
   - outlier detection (Z-Score, IQR, MAD, Winsorization, Isolation Forest)
   - normalization (Z-Score, Min-Max, RobustScaler)
4. Compare methods via interactive plots.
5. Apply the chosen preprocessing pipeline.
6. Export cleaned data as CSV.

---------------------------------------
METHODOLOGY
---------------------------------------
CIRRUS implements domain-specific preprocessing heuristics based on prior empirical studies. Key guidelines include:

- Low missingness (<5%): LOCF or Mean Imputation
- Moderate missingness (5–20%): KNN or MICE
- Gaussian-like distributions with few outliers: Z-Score
- Heavy skew or high outlier rate (>15%): MAD or Isolation Forest
- Normalization chosen by distribution shape:
  Z-Score for near-Gaussian data
  Min-Max for gaze features
  RobustScaler for pupil or heavy-tailed features

These rules are organized into a three-dimensional decision space reflecting the interaction between missingness, outliers, and skewness.

---------------------------------------
EXTENSIBILITY
---------------------------------------
CIRRUS is implemented using modular helper functions for imputation, outlier detection, normalization, visualization, and heuristic decisions. New preprocessing techniques or metrics can be integrated by extending these components. The functions can also be reused in external analysis workflows or scikit-learn pipelines.

---------------------------------------
LICENSE
---------------------------------------
no

---------------------------------------
CONTACT
---------------------------------------
Jennifer Landes
University of Regensburg
Email: jennifer.landes@informatik.uni-regensburg.de
