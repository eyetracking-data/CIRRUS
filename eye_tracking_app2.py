# Eye-Tracking Data Analysis App with Complete Preprocessing and Recommendations
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft
from scipy.stats import zscore, median_abs_deviation, skew  # kurtosis via pandas
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import plotly.graph_objs as go  # interactive plots


st.set_page_config(page_title="Eye-Tracking Preprocessing Tool", layout="wide")

st.title("👁️ Eye-Tracking Data Preprocessing & Recommendations")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")


# ---------- HELPER FUNCTIONS & HEURISTICS ----------

def apply_normalization(norm_input, method):
    """
    norm_input: np.ndarray shape (n, 1)
    method: string from ["None", "StandardScaler (Z-Score)", "RobustScaler", "MinMaxScaler"]
    """
    if method == "StandardScaler (Z-Score)":
        mean_val = norm_input.mean()
        std_val = norm_input.std()
        if std_val == 0 or np.isnan(std_val):
            # Degenerate case: center only, no scaling
            return norm_input - mean_val
        return (norm_input - mean_val) / std_val

    elif method == "RobustScaler":
        q1 = np.quantile(norm_input, 0.25)
        q3 = np.quantile(norm_input, 0.75)
        denom = (q3 - q1)
        if denom == 0 or np.isnan(denom):
            # Degenerate case: center on q1
            return norm_input - q1
        return (norm_input - q1) / denom

    elif method == "MinMaxScaler":
        min_val = norm_input.min()
        max_val = norm_input.max()
        denom = (max_val - min_val)
        if denom == 0 or np.isnan(denom):
            # Degenerate case: shift to zero
            return norm_input - min_val
        return (norm_input - min_val) / denom

    else:  # "None" or unknown
        return norm_input


def validate_single_method(selection, name="method"):
    if len(selection) > 1:
        st.warning(f"⚠️ Please select only **one** {name} to proceed.")
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


# --- Heuristic: Missing Values ---
def get_missing_value_recommendation(missing_rate):
    txt = f"Missing rate: **{missing_rate:.1f}%**.\n\n"

    if missing_rate < 5:
        txt += (
            "🔹 **Recommended:** LOCF (for short gaps in time series) or Mean Imputation.\n"
            "🔹 **Alternatives:** KNN Imputation if you want to preserve multivariate structure.\n"
            "⚠️ Complex methods such as MICE usually don't pay off here."
        )
    elif missing_rate <= 10:
        txt += (
            "🔹 **Recommended:** KNN Imputation (best trade-off around 5–10% missing).\n"
            "🔹 **Alternatives:**\n"
            "   • LOCF if the gaps are short and temporally local.\n"
            "   • MICE if you expect complex/nonlinear relationships.\n"
        )
    elif missing_rate <= 20:
        txt += (
            "🔹 **Recommended:** KNN **or** MICE (5–20% missing → robust, model-based imputation needed).\n"
            "🔹 **Alternatives:**\n"
            "   • Keep the feature but interpret downstream analyses with caution.\n"
            "   • Only seriously consider feature removal beyond ~30–40% missing."
        )
    else:
        txt += (
            "🔹 **Recommended:** KNN or MICE, but overall data quality is questionable.\n"
            "🔹 **Alternatives:**\n"
            "   • Drop the feature if missingness is extremely high (e.g. >40%).\n"
            "   • Check whether a sensor/channel fails systematically (design issue rather than preprocessing)."
        )
    return txt


# --- Heuristic: Outlier Methods ---
def get_outlier_recommendation(outlier_rate, skew_val):
    txt = f"Estimated outlier rate (IQR-based): **{outlier_rate:.1f}%**. Skewness: **{skew_val:.2f}**.\n\n"
    gaussian = abs(skew_val) < 0.5
    highly_skewed = abs(skew_val) > 1.0

    if outlier_rate < 5 and gaussian:
        txt += (
            "🔹 **Recommended:** Z-Score (fast, interpretable, good for quasi-Gaussian data).\n"
            "🔹 **Alternatives:** IQR (robust) or Isolation Forest if you later use nonlinear models."
        )
    elif outlier_rate <= 15:
        if highly_skewed:
            txt += (
                "🔹 **Recommended:** MAD **or** Isolation Forest (robust under strong skew).\n"
                "🔹 **Alternatives:** Winsorization for visualization; avoid Z-Score here."
            )
        else:
            txt += (
                "🔹 **Recommended:** IQR or Isolation Forest (solid choice with 5–15% outliers).\n"
                "🔹 **Alternatives:** MAD for heavier tails; Winsorization for clean plots."
            )
    else:
        txt += (
            "🔹 **Recommended:** Isolation Forest (very high outlier load, non-parametric).\n"
            "🔹 **Alternatives:** MAD for univariate cleaning plus Winsorization if you must keep sample size."
        )
    return txt


# --- Heuristic: Normalization ---
def get_normalization_recommendation(skew_val, outlier_rate):
    txt = f"Skewness: **{skew_val:.2f}**, outlier rate: **{outlier_rate:.1f}%**.\n\n"

    if abs(skew_val) < 0.5 and outlier_rate < 10:
        txt += (
            "🔹 **Recommended:** StandardScaler (Z-Score), good for many ML models with roughly normal data.\n"
            "🔹 **Alternatives:**\n"
            "   • Min-Max scaling for visualization or tree-based models (RF, Isolation Forest).\n"
            "   • RobustScaler only if you detect more outliers later."
        )
    elif outlier_rate > 15 or abs(skew_val) > 1.0:
        txt += (
            "🔹 **Recommended:** RobustScaler (robust to outliers and skew, often good for pupil features).\n"
            "🔹 **Alternatives:**\n"
            "   • Log/power transform + StandardScaler for heavy right skew.\n"
            "   • Min-Max only after aggressive outlier handling."
        )
    else:
        txt += (
            "🔹 **Recommended:** Min-Max scaling (good for many gaze features, visualization, RF + IF).\n"
            "🔹 **Alternatives:**\n"
            "   • RobustScaler for pupil/intensity-like features.\n"
            "   • StandardScaler if your downstream ML pipeline expects it."
        )
    return txt


# --- Heuristic: Full Pipeline (Missing + Outliers) ---
def get_pipeline_recommendation(missing_rate, outlier_rate):
    txt = f"Pipeline heuristic based on missing: **{missing_rate:.1f}%**, outliers: **{outlier_rate:.1f}%**.\n\n"

    if missing_rate < 5 and outlier_rate < 10:
        txt += (
            "🔹 **Recommended pipeline:** **LOCF + Z-Score + optional smoothing**.\n"
            "   • LOCF maintains time-series continuity for short gaps.\n"
            "   • Z-Score is sufficient for moderate, quasi-Gaussian distributions.\n"
            "   • Apply smoothing (Butterworth / Fourier) only if high-frequency noise is problematic."
        )
    elif 5 <= missing_rate <= 10 and 10 <= outlier_rate <= 15:
        txt += (
            "🔹 **Recommended pipeline:** **KNN + Isolation Forest + Min-Max scaling**.\n"
            "   • KNN uses multivariate structure up to ~10% missing.\n"
            "   • Isolation Forest is robust for 10–15% outliers.\n"
            "   • Min-Max scaling works well with RF/IF-based pipelines."
        )
    elif missing_rate > 10 and outlier_rate > 15:
        txt += (
            "🔹 **Recommended pipeline:** **MICE or KNN + MAD or Isolation Forest + RobustScaler**.\n"
            "   • MICE/KNN for complex missingness patterns.\n"
            "   • MAD / Isolation Forest for robust outlier handling.\n"
            "   • RobustScaler for stable scaling under heavy outlier burden.\n"
            "⚠️ Critically check data quality – consider dropping features with extreme issues."
        )
    else:
        txt += (
            "🔹 **Mixed case:**\n"
            "   • Missing: KNN or MICE depending on complexity.\n"
            "   • Outliers: Isolation Forest or MAD (depending on skewness).\n"
            "   • Scaling: Min-Max for moderate outliers, otherwise RobustScaler.\n"
            "Use the detailed recommendations in each section as guidance."
        )
    return txt


# ---------- MAIN APP ----------

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

        # -------- METRICS (Missing, Outliers, Skew, Kurtosis) --------
        feature_series = original_data[selected_feature_plot]
        missing_rate_feature = feature_series.isna().mean() * 100

        non_na = feature_series.dropna()
        if len(non_na) > 1:
            # Outlier rate via IQR
            Q1, Q3 = np.percentile(non_na, [25, 75])
            IQR_val = Q3 - Q1 if Q3 > Q1 else 0.0
            if IQR_val > 0:
                lower = Q1 - 1.5 * IQR_val
                upper = Q3 + 1.5 * IQR_val
                outliers_mask = (non_na < lower) | (non_na > upper)
                outlier_rate_feature = outliers_mask.mean() * 100
            else:
                outlier_rate_feature = 0.0

            skew_val = skew(non_na)
            kurt_val = non_na.kurt()  # pandas kurtosis (Fisher)
        else:
            outlier_rate_feature = 0.0
            skew_val = 0.0
            kurt_val = 0.0

        st.subheader("📊 Feature Diagnostics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing Rate", f"{missing_rate_feature:.1f}%")
        c2.metric("Outlier Rate (IQR)", f"{outlier_rate_feature:.1f}%")
        c3.metric("Skewness", f"{skew_val:.2f}")
        c4.metric("Kurtosis", f"{kurt_val:.2f}")

        st.info(get_pipeline_recommendation(missing_rate_feature, outlier_rate_feature))

        # ---------- RAW ORIGINAL FEATURE PLOT ----------

        st.subheader("📈 Original Feature Plot")

        fig_raw = go.Figure()
        fig_raw.add_trace(
            go.Scatter(
                x=feature_series.index,
                y=feature_series.values,
                mode="lines",
                name="Original",
                line=dict(color="black", width=2)
            )
        )
        fig_raw.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False
        )
        st.plotly_chart(fig_raw, use_container_width=True)

        # ---------- MISSING VALUES: PREVIEW + RECOMMENDATIONS ----------

        mv_methods = ["Mean Imputation", "LOCF", "Linear Interpolation", "KNN Imputation", "MICE"]

        st.subheader("🧪 Missing Value Strategy Preview")
        preview_mv_methods = st.multiselect("Select Missing Value Methods to Compare", mv_methods)

        # Interactive Plot with Plotly
        fig_mv = go.Figure()

        # Original line – clear, dashed
        orig_series = original_data[selected_feature_plot]
        fig_mv.add_trace(
            go.Scatter(
                x=orig_series.index,
                y=orig_series.values,
                mode="lines",
                name="Original",
                line=dict(color="black", width=2, dash="dash")
            )
        )

        # Methods – thinner, slightly transparent
        colors = ["blue", "green", "red", "purple", "orange"]
        for i, method in enumerate(preview_mv_methods):
            preview_data = apply_missing_value_method(
                original_data[[selected_feature_plot]].copy(), method
            )[selected_feature_plot]
            fig_mv.add_trace(
                go.Scatter(
                    x=preview_data.index,
                    y=preview_data.values,
                    mode="lines",
                    name=method,
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.7
                )
            )

        fig_mv.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        st.plotly_chart(fig_mv, use_container_width=True)

        # Recommendation based on heuristic table
        st.caption("Recommendation (Missing Values):")
        st.info(get_missing_value_recommendation(missing_rate_feature))

        # Final MV selection to apply
        final_mv_method = st.selectbox("✅ Apply this MV method for next steps", ["None"] + mv_methods)
        data_mv_applied = apply_missing_value_method(data.copy(), final_mv_method) if final_mv_method != "None" else data.copy()

        # ---------- OUTLIER DETECTION: PREVIEW + RECOMMENDATIONS ----------

        st.subheader("🧢 Preview: Outlier Detection")
        outlier_methods = ["Z-Score", "IQR", "MAD", "Winsorization", "Isolation Forest"]
        preview_outlier_methods = st.multiselect("Select Outlier Methods to Compare", outlier_methods)
        selected_outlier_method = st.selectbox("✅ Apply this Outlier Method", ["None"] + outlier_methods)

        z_thresh = st.slider("Z-Score threshold", 1.0, 5.0, 3.0)
        contamination = st.slider("Isolation Forest contamination", 0.01, 0.5, 0.1)

        # Recommendation for outlier methods
        st.caption("Recommendation (Outliers):")
        st.info(get_outlier_recommendation(outlier_rate_feature, skew_val))

        x = data_mv_applied[selected_feature_plot]

        with st.expander("🔎 Outlier Preview (Original + Methods)", expanded=True):
            fig_out = go.Figure()

            # Original
            fig_out.add_trace(
                go.Scatter(
                    x=x.index,
                    y=x.values,
                    mode="lines",
                    name="Original",
                    line=dict(color="black", width=2, dash="dash")
                )
            )

            colors_out = ["blue", "green", "red", "purple", "orange"]
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
                    preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(
                        temp.fillna(temp.mean()).values.reshape(-1, 1)
                    )
                    outliers = preds == -1
                    temp[outliers] = np.nan

                fig_out.add_trace(
                    go.Scatter(
                        x=temp.index,
                        y=temp.values,
                        mode="lines",
                        name=method,
                        line=dict(color=colors_out[i % len(colors_out)], width=1),
                        opacity=0.7
                    )
                )

            fig_out.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )

            st.plotly_chart(fig_out, use_container_width=True)

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
            preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(
                outlier_data.fillna(outlier_data.mean()).values.reshape(-1, 1)
            )
            outliers = preds == -1
            outlier_data[outliers] = np.nan

        # ---------- NORMALIZATION: PREVIEW + RECOMMENDATIONS ----------

        st.subheader("📐 Preview: Normalization")

        with st.expander("🔧 Manual Y-Axis Zoom"):
            y_min_manual = st.number_input("Y-Axis Min (optional)", value=-1.0, step=0.1, format="%.2f")
            y_max_manual = st.number_input("Y-Axis Max (optional)", value=2.0, step=0.1, format="%.2f")
            custom_ylim = st.checkbox("Use custom Y-axis range")

        norm_methods = ["StandardScaler (Z-Score)", "RobustScaler", "MinMaxScaler"]
        preview_norm_methods = st.multiselect("Select normalization methods to preview", norm_methods)

        # Base: values after outlier handling
        norm_input = outlier_data.fillna(method="ffill").values.reshape(-1, 1)
        norm_index = outlier_data.index

        fig_norm = go.Figure()

        # Original line
        fig_norm.add_trace(
            go.Scatter(
                x=norm_index,
                y=norm_input.flatten(),
                mode="lines",
                name="Original",
                line=dict(color="black", width=2, dash="dash")
            )
        )

        colors_norm = ["blue", "green", "red"]

        # PREVIEW ONLY – no influence on pipeline
        for i, method in enumerate(preview_norm_methods):
            tmp = apply_normalization(norm_input, method)
            fig_norm.add_trace(
                go.Scatter(
                    x=norm_index,
                    y=tmp.flatten(),
                    mode="lines",
                    name=method,
                    line=dict(color=colors_norm[i % len(colors_norm)], width=1),
                    opacity=0.7
                )
            )

        # Y-range (optional)
        if preview_norm_methods and custom_ylim:
            fig_norm.update_yaxes(range=[y_min_manual, y_max_manual])

        fig_norm.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        st.plotly_chart(fig_norm, use_container_width=True)

        # Recommendation for normalization
        st.caption("Recommendation (Normalization):")
        st.info(get_normalization_recommendation(skew_val, outlier_rate_feature))

        # This actually decides what goes into the pipeline
        selected_norm_method = st.selectbox("✅ Apply this Normalization Method", ["None"] + norm_methods)

        # Normalization for the pipeline (without optional filters)
        norm_output_base = apply_normalization(norm_input, selected_norm_method)

        # ---------- SHORT SUMMARY OF APPLIED METHODS ----------

        st.subheader("✅ Applied Preprocessing Summary (before optional smoothing)")
        st.markdown(
            f"- Missing value method: **{final_mv_method if final_mv_method != 'None' else 'None'}**  \n"
            f"- Outlier method: **{selected_outlier_method if selected_outlier_method != 'None' else 'None'}**  \n"
            f"- Normalization: **{selected_norm_method if selected_norm_method != 'None' else 'None'}**"
        )

        # Plot: Original vs. applied pipeline (MV + outliers + normalization)
        applied_series = pd.Series(norm_output_base.flatten(), index=norm_index)

        fig_applied = go.Figure()
        fig_applied.add_trace(
            go.Scatter(
                x=feature_series.index,
                y=feature_series.values,
                mode="lines",
                name="Original",
                line=dict(color="gray", width=1, dash="dash")
            )
        )
        fig_applied.add_trace(
            go.Scatter(
                x=applied_series.index,
                y=applied_series.values,
                mode="lines",
                name="After MV + Outliers + Normalization",
                line=dict(color="blue", width=2)
            )
        )
        fig_applied.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_applied, use_container_width=True)

        # ---------- OPTIONAL POST-NORMALIZATION (visual only) ----------

        st.subheader("⚙️ Optional Post-Normalization Steps")
        st.caption(
            "These steps act on the **applied** normalization (blue line above). "
            "They are primarily for visual smoothing and are **not** included in the CSV export."
        )

        # Base series for optional steps – explicit float
        preview_series = applied_series.astype(float)

        # Fallback: if everything is NaN (e.g. too aggressive outlier removal), use raw feature for visualization
        if np.all(np.isnan(preview_series.values)):
            st.warning(
                "All values after normalization are NaN – using the original feature as a fallback for optional visual steps. "
                "Check the selected outlier/normalization settings if this is unexpected."
            )
            preview_series = feature_series.astype(float)

        # 1) Manual limits
        if st.checkbox("Apply manual feature limits"):
            if len(preview_series) < 2:
                st.warning("Too few data points for limit visualization.")
            else:
                min_val = st.number_input("Lower limit", value=float(np.nanmin(preview_series)))
                max_val = st.number_input("Upper limit", value=float(np.nanmax(preview_series)))

                clipped_series = preview_series.clip(lower=min_val, upper=max_val)

                fig_clip = go.Figure()
                fig_clip.add_trace(
                    go.Scatter(
                        x=preview_series.index,
                        y=preview_series.values,
                        mode="lines",
                        name="Before Limits",
                        line=dict(width=1, dash="dash")
                    )
                )
                fig_clip.add_trace(
                    go.Scatter(
                        x=clipped_series.index,
                        y=clipped_series.values,
                        mode="lines",
                        name="After Limits",
                        line=dict(width=2)
                    )
                )

                fig_clip.update_layout(
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig_clip, use_container_width=True)

                # update base for following steps
                preview_series = clipped_series

        # 2) Butterworth low-pass filter
        if st.checkbox("Apply Butterworth low-pass filter"):
            if len(preview_series) < 4:
                st.warning("Too few data points for Butterworth filter.")
            else:
                cutoff = st.slider("Cutoff Frequency", 0.01, 0.5, 0.1)
                fs = st.number_input("Sampling Frequency (Hz)", value=1.0)
                order = st.slider("Filter Order", 1, 10, 2)
                try:
                    signal = preview_series.to_numpy(dtype=float)
                    filtered = butter_lowpass_filter(signal, cutoff, fs, order)
                    filtered_series = pd.Series(filtered, index=preview_series.index)

                    fig_filt = go.Figure()
                    fig_filt.add_trace(
                        go.Scatter(
                            x=preview_series.index,
                            y=preview_series.values,
                            mode="lines",
                            name="Before Filter",
                            line=dict(width=1, dash="dash")
                        )
                    )
                    fig_filt.add_trace(
                        go.Scatter(
                            x=filtered_series.index,
                            y=filtered_series.values,
                            mode="lines",
                            name="After Filter",
                            line=dict(width=2)
                        )
                    )
                    fig_filt.update_layout(
                        height=250,
                        margin=dict(l=40, r=20, t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                    )
                    st.plotly_chart(fig_filt, use_container_width=True)

                    preview_series = filtered_series
                except Exception as e:
                    st.error(f"Filter error: {e}")

        # 3) Fourier smoothing
        if st.checkbox("Apply Fourier Smoothing"):
            if len(preview_series) < 4:
                st.warning("Too few data points for Fourier smoothing.")
            else:
                freq_limit = st.slider("Max frequency to keep (Hz)", 0.01, 0.5, 0.1)
                signal = preview_series.to_numpy(dtype=float)
                fft_vals = fft(signal)
                freq = np.fft.fftfreq(len(fft_vals))
                fft_vals[np.abs(freq) > freq_limit] = 0
                smoothed = np.real(ifft(fft_vals))
                smoothed_series = pd.Series(smoothed, index=preview_series.index)

                fig_fft = go.Figure()
                fig_fft.add_trace(
                    go.Scatter(
                        x=preview_series.index,
                        y=preview_series.values,
                        mode="lines",
                        name="Before Fourier",
                        line=dict(width=1, dash="dash")
                    )
                )
                fig_fft.add_trace(
                    go.Scatter(
                        x=smoothed_series.index,
                        y=smoothed_series.values,
                        mode="lines",
                        name="After Fourier",
                        line=dict(width=2)
                    )
                )
                fig_fft.update_layout(
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig_fft, use_container_width=True)

                preview_series = smoothed_series

        # ---------- Final normalized data for export (applied normalization, no optional filters) ----------

        norm_output = norm_output_base.copy()

        # ---------- FINAL COMPARISON & EXPORT ----------

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

