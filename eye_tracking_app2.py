# Eye-Tracking Data Analysis App with Complete Preprocessing and Recommendations
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft
from scipy.stats import zscore, iqr, median_abs_deviation, skew
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Eye-Tracking Preprocessing Tool", layout="wide")

st.title("👁️ Eye-Tracking Data Preprocessing & Recommendations")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

def validate_single_method(selection, name="Methode"):
    if len(selection) > 1:
        st.warning(f"⚠️ Bitte nur **eine** {name} auswählen, um fortzufahren.")
        return None
    elif len(selection) == 0:
        return None
    return selection[0]

def apply_missing_value_method(data, method):
    if method == "Mean Imputation":
        return data.fillna(data.mean())
    elif method == "LOCF":
        return data.fillna(method="ffill")
    elif method == "Linear Interpolation":
        return data.interpolate()
    elif method == "KNN Imputation":
        return pd.DataFrame(KNNImputer().fit_transform(data), columns=data.columns)
    elif method == "MICE":
        return pd.DataFrame(IterativeImputer().fit_transform(data), columns=data.columns)
    return data

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File successfully uploaded.")

    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Select Features to Analyze")
    features = st.multiselect("Choose numerical features", df.select_dtypes(include=[np.number]).columns.tolist())

    if features:
        data = df[features].copy()
        original_data = data.copy()

        selected_feature_plot = st.selectbox("🔍 Select Feature for Preview", features)

        mv_methods = ["Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"]
        preview_mv_methods = st.multiselect("Select Missing Value Methods to Compare", mv_methods)
        st.subheader("🧪 Missing Value Strategy Preview")
        fig, ax = plt.subplots()
        ax.plot(original_data[selected_feature_plot], label="Original", linestyle='--', color='gray')
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, method in enumerate(preview_mv_methods):
            preview_data = apply_missing_value_method(original_data[[selected_feature_plot]].copy(), method)
            ax.plot(preview_data, label=method, color=colors[i % len(colors)], alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        # Final MV selection to apply
        final_mv_method = st.selectbox("✅ Apply this MV method for next steps", ["None"] + mv_methods)
        data_mv_applied = apply_missing_value_method(data.copy(), final_mv_method) if final_mv_method != "None" else data.copy()

        missing_rate = data.isnull().mean() * 100
        st.caption("Recommendation:")
        if missing_rate.mean() < 5:
            st.info("Missing Values: Recommended → Mean or LOCF (<5% missing)")
        elif missing_rate.mean() <= 10:
            st.info("Missing Values: Recommended → KNN Imputation (5–10% missing)")
        elif missing_rate.mean() <= 20:
            st.info("Missing Values: Recommended → KNN or MICE (>10% missing)")
        else:
            st.warning("Missing Values: High missingness – consider feature removal")

        st.subheader("🧢 Preview: Outlier Detection")
        outlier_methods = ["Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"]
        preview_outlier_methods = st.multiselect("Select Outlier Methods to Compare", outlier_methods)
        selected_outlier_method = st.selectbox("✅ Apply this Outlier Method", ["None"] + outlier_methods)

        z_thresh = st.slider("Z-Score threshold", 1.0, 5.0, 3.0)
        contamination = st.slider("Isolation Forest contamination", 0.01, 0.5, 0.1)

        fig, ax = plt.subplots()
        x = data_mv_applied[selected_feature_plot]
        ax.plot(x.index, x.values, label="Original", linestyle='--', color='black')

        for i, method in enumerate(preview_outlier_methods):
            temp = x.copy()
            if method == "Z-Score":
                outliers = np.abs(zscore(temp.fillna(temp.mean()))) > z_thresh
                temp[outliers] = np.nan
            elif method == "IQR":
                Q1, Q3 = np.percentile(temp.dropna(), [25, 75])
                IQR_val = Q3 - Q1
                outliers = (temp < Q1 - 1.5 * IQR_val) | (temp > Q3 + 1.5 * IQR_val)
                temp[outliers] = np.nan
            elif method == "MAD":
                med = np.median(temp.dropna())
                mad = median_abs_deviation(temp.dropna())
                outliers = np.abs(temp - med) > 3 * mad
                temp[outliers] = np.nan
            elif method == "Winsorization":
                lower = temp.quantile(0.05)
                upper = temp.quantile(0.95)
                temp = np.clip(temp, lower, upper)
            elif method == "Isolation Forest":
                preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(temp.fillna(temp.mean()).values.reshape(-1, 1))
                outliers = preds == -1
                temp[outliers] = np.nan
            ax.plot(temp, label=method, alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        # Apply selected outlier method
        outlier_data = data_mv_applied[selected_feature_plot].copy()
        if selected_outlier_method == "Z-Score":
            outliers = np.abs(zscore(outlier_data.fillna(outlier_data.mean()))) > z_thresh
            outlier_data[outliers] = np.nan
        elif selected_outlier_method == "IQR":
            Q1, Q3 = np.percentile(outlier_data.dropna(), [25, 75])
            IQR_val = Q3 - Q1
            outliers = (outlier_data < Q1 - 1.5 * IQR_val) | (outlier_data > Q3 + 1.5 * IQR_val)
            outlier_data[outliers] = np.nan
        elif selected_outlier_method == "MAD":
            med = np.median(outlier_data.dropna())
            mad = median_abs_deviation(outlier_data.dropna())
            outliers = np.abs(outlier_data - med) > 3 * mad
            outlier_data[outliers] = np.nan
        elif selected_outlier_method == "Winsorization":
            lower = outlier_data.quantile(0.05)
            upper = outlier_data.quantile(0.95)
            outlier_data = np.clip(outlier_data, lower, upper)
        elif selected_outlier_method == "Isolation Forest":
            preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(outlier_data.fillna(outlier_data.mean()).values.reshape(-1, 1))
            outliers = preds == -1
            outlier_data[outliers] = np.nan

        # Removed duplicate chart to avoid double plot

        # Normalization Preview
        st.subheader("📐 Preview: Normalization")

        with st.expander("🔧 Manual Y-Axis Zoom"):
            y_min_manual = st.number_input("Y-Axis Min (optional)", value=-1.0, step=0.1, format="%.2f")
            y_max_manual = st.number_input("Y-Axis Max (optional)", value=2.0, step=0.1, format="%.2f")
            custom_ylim = st.checkbox("Use custom Y-axis range")
        norm_methods = ["StandardScaler (Z-Score)", "RobustScaler", "MinMaxScaler"]
        preview_norm_methods = st.multiselect("Select normalization methods to preview", norm_methods)

        fig, ax = plt.subplots()
        norm_input = outlier_data.fillna(method="ffill").values.reshape(-1, 1)
        ax.plot(norm_input, label="Original", linestyle='--', color='gray')

        for i, method in enumerate(preview_norm_methods):
            if method == "StandardScaler (Z-Score)":
                norm_output = (norm_input - norm_input.mean()) / norm_input.std()
            elif method == "RobustScaler":
                q1 = np.quantile(norm_input, 0.25)
                q3 = np.quantile(norm_input, 0.75)
                norm_output = (norm_input - q1) / (q3 - q1)
            elif method == "MinMaxScaler":
                norm_output = (norm_input - norm_input.min()) / (norm_input.max() - norm_input.min())
            ax.plot(norm_output, label=method, alpha=0.7)

        if preview_norm_methods:
            if custom_ylim:
                ax.set_ylim(y_min_manual, y_max_manual)
            else:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin - 0.2 * abs(ymin), ymax + 0.2 * abs(ymax))

        ax.legend()
        st.pyplot(fig)

        selected_norm_method = st.selectbox("✅ Apply this Normalization Method", ["None"] + norm_methods)

        # Optional post-normalization steps
        st.subheader("⚙️ Optional Post-Normalization Steps")

        if st.checkbox("Apply manual feature limits"):
            min_val = st.number_input("Lower limit", value=float(np.min(norm_output)))
            max_val = st.number_input("Upper limit", value=float(np.max(norm_output)))
            clipped_output = np.clip(norm_output, min_val, max_val)
            fig, ax = plt.subplots()
            ax.plot(norm_output, label="Before Limits", linestyle='--', alpha=0.5)
            ax.plot(clipped_output, label="After Limits", color='green')
            ax.axhline(min_val, color='red', linestyle=':')
            ax.axhline(max_val, color='red', linestyle=':')
            ax.legend()
            st.pyplot(fig)
            norm_output = clipped_output

        if st.checkbox("Apply Butterworth low-pass filter"):
            cutoff = st.slider("Cutoff Frequency", 0.01, 0.5, 0.1)
            fs = st.number_input("Sampling Frequency (Hz)", value=1.0)
            order = st.slider("Filter Order", 1, 10, 2)
            try:
                norm_output = butter_lowpass_filter(norm_output.flatten(), cutoff, fs, order)
                st.line_chart(norm_output, height=200)
            except Exception as e:
                st.error(f"Filter error: {e}")

        if st.checkbox("Apply Fourier Smoothing"):
            freq_limit = st.slider("Max frequency to keep (Hz)", 0.01, 0.5, 0.1)
            fft_vals = fft(norm_output.flatten())
            freq = np.fft.fftfreq(len(fft_vals))
            fft_vals[np.abs(freq) > freq_limit] = 0
            smoothed = np.real(ifft(fft_vals))
            st.line_chart(smoothed, height=200)
            norm_output = smoothed

        # Final normalized data for output
        if selected_norm_method == "StandardScaler (Z-Score)":
            norm_output = (norm_input - norm_input.mean()) / norm_input.std()
        elif selected_norm_method == "RobustScaler":
            q1 = np.quantile(norm_input, 0.25)
            q3 = np.quantile(norm_input, 0.75)
            norm_output = (norm_input - q1) / (q3 - q1)
        elif selected_norm_method == "MinMaxScaler":
            norm_output = (norm_input - norm_input.min()) / (norm_input.max() - norm_input.min())
        else:
            norm_output = norm_input

        # Final Comparison
        if st.button("🔎 Show Final Comparison"):
            st.subheader("📊 Summary Statistics Before vs After")

            final_series = norm_output.flatten()
            before_series = original_data[selected_feature_plot].copy()

            missing_before = before_series.isnull().mean() * 100
            missing_after = np.isnan(final_series).mean() * 100

            outlier_removed = np.count_nonzero(before_series.notna()) - np.count_nonzero(~np.isnan(outlier_data))
            outlier_percent = (outlier_removed / len(before_series)) * 100

            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Std", "Missing %", "Outlier Removed %"],
                "Before": [before_series.mean(), before_series.std(), missing_before, 0.0],
                "After": [np.nanmean(final_series), np.nanstd(final_series), missing_after, outlier_percent]
            })
            st.table(stats_df)

            st.subheader("📋 Final Data Preview")
            final_data = df.copy()
            final_data[selected_feature_plot] = final_series
            st.dataframe(final_data.head())

            csv = final_data.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Processed CSV", csv, file_name="processed_data.csv")
