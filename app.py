import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import cv2


# -----------------------------
# Load Model
# -----------------------------
model = load_model("ecg_multiclass_model.h5")
class_names = ["Normal", "PVC", "AF", "Arrhythmia"]


# -----------------------------
# ECG Image to Signal Function
# -----------------------------
def extract_signal_from_ecg_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    h, w = thresh.shape

    roi = thresh[
        int(h * 0.25):int(h * 0.85),
        int(w * 0.05):int(w * 0.95)
    ]

    signal = []

    for x in range(roi.shape[1]):
        ys = np.where(roi[:, x] > 0)[0]

        if len(ys) > 0:
            y = np.mean(ys)
            signal.append(y)
        else:
            signal.append(np.nan)

    signal = np.array(signal)

    nans = np.isnan(signal)

    if np.any(nans):
        signal[nans] = np.interp(
            np.flatnonzero(nans),
            np.flatnonzero(~nans),
            signal[~nans]
        )

    signal = -signal

    signal = np.interp(
        np.linspace(0, len(signal) - 1, 200),
        np.arange(len(signal)),
        signal
    )

    signal = (signal - np.min(signal)) / (
        np.max(signal) - np.min(signal)
    )

    return signal


# -----------------------------
# Risk Function
# -----------------------------
def get_risk_level(result):
    if result == "Normal":
        return "Low Risk"
    elif result == "PVC":
        return "Medium Risk"
    elif result == "AF":
        return "High Risk"
    else:
        return "Critical Risk"


# -----------------------------
# Single Beat Feature Explanation
# -----------------------------
def extract_ecg_features(signal, predicted_result):
    signal = np.array(signal).flatten()

    features = {
        "heart_rate": "Estimated",
        "heart_rate_impact": 0,
        "qrs_width": "Estimated",
        "qrs_width_impact": 0,
        "rr_interval": "Single beat input",
        "rr_impact": 0
    }

    # Estimate QRS Width from one beat
    try:
        mean_val = np.mean(signal)
        max_val = np.max(signal)

        threshold = mean_val + 0.6 * (max_val - mean_val)

        qrs_points = np.where(signal > threshold)[0]

        if len(qrs_points) > 0:
            qrs_width_samples = qrs_points[-1] - qrs_points[0]

            qrs_width_ms = round((qrs_width_samples / 200) * 1000, 2)

            features["qrs_width"] = qrs_width_ms

            if qrs_width_ms > 120:
                features["qrs_width_impact"] = 40
            elif qrs_width_ms > 100:
                features["qrs_width_impact"] = 25
            else:
                features["qrs_width_impact"] = 10

    except:
        pass

    # Class-based explanation rules
    if predicted_result == "Normal":
        features["heart_rate"] = "Normal range"
        features["heart_rate_impact"] = 5
        features["rr_interval"] = "Regular"
        features["rr_impact"] = 5

    elif predicted_result == "PVC":
        features["heart_rate"] = "May be elevated"
        features["heart_rate_impact"] = 30
        features["qrs_width_impact"] = max(features["qrs_width_impact"], 40)
        features["rr_interval"] = "Irregular beat pattern"
        features["rr_impact"] = 20

    elif predicted_result == "AF":
        features["heart_rate"] = "Often irregular/elevated"
        features["heart_rate_impact"] = 30
        features["rr_interval"] = "Irregular RR rhythm"
        features["rr_impact"] = 20
        features["qrs_width_impact"] = max(features["qrs_width_impact"], 15)

    elif predicted_result == "Arrhythmia":
        features["heart_rate"] = "Abnormal rhythm suspected"
        features["heart_rate_impact"] = 30
        features["rr_interval"] = "Irregular rhythm"
        features["rr_impact"] = 20
        features["qrs_width_impact"] = max(features["qrs_width_impact"], 30)

    return features


# -----------------------------
# Display Feature Explanation
# -----------------------------
def show_feature_explanation(signal, predicted_result):
    st.subheader("ECG Feature Explanation")

    features = extract_ecg_features(signal, predicted_result)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Heart Rate",
            f"{features['heart_rate']}",
            f"+{features['heart_rate_impact']}%"
        )
        st.progress(features["heart_rate_impact"] / 100)

    with col2:
        st.metric(
            "QRS Width",
            f"{features['qrs_width']} ms",
            f"+{features['qrs_width_impact']}%"
        )
        st.progress(features["qrs_width_impact"] / 100)

    with col3:
        st.metric(
            "RR Interval",
            f"{features['rr_interval']}",
            f"+{features['rr_impact']}%"
        )
        st.progress(features["rr_impact"] / 100)

    total_impact = (
        features["heart_rate_impact"]
        + features["qrs_width_impact"]
        + features["rr_impact"]
    )

    st.info(
        f"""
        Explanation Summary:

        Heart Rate = {features['heart_rate']}  
        Impact = +{features['heart_rate_impact']}%

        QRS Width = {features['qrs_width']} ms  
        Impact = +{features['qrs_width_impact']}%

        RR Interval = {features['rr_interval']}  
        Impact = +{features['rr_impact']}%

        Total Explanation Score = {total_impact}%

        Note: This ECG input contains only one beat. Therefore, Heart Rate and RR Interval are estimated using prediction-based explanation rules.
        """
    )


# -----------------------------
# Main Dashboard
# -----------------------------
st.title("ECG Explainable AI Dashboard")

tab1, tab2 = st.tabs(
    [
        "ECG Signal Upload (.npy)",
        "ECG Image Upload"
    ]
)


# -----------------------------
# TAB 1: NPY ECG Signal Upload
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload ECG Beat (.npy)",
        type=["npy"]
    )

    if uploaded_file is not None:
        signal = np.load(uploaded_file)

        signal_1d = signal.reshape(200)
        signal_input = signal_1d.reshape(1, 200, 1)

        prediction = model.predict(signal_input)[0]

        predicted_class = np.argmax(prediction)
        result = class_names[predicted_class]
        confidence = prediction[predicted_class] * 100
        risk = get_risk_level(result)

        if result == "Normal":
            st.success(f"Prediction: {result}")
            st.success(f"Risk Level: {risk}")
        elif result == "PVC":
            st.warning(f"Prediction: {result}")
            st.warning(f"Risk Level: {risk}")
        else:
            st.error(f"Prediction: {result}")
            st.error(f"Risk Level: {risk}")

        st.write(f"Confidence: {confidence:.2f}%")

        importance = np.abs(signal_1d - np.mean(signal_1d))
        mask = importance > np.percentile(importance, 85)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal_1d, label="ECG Beat")
        ax.fill_between(
            range(len(signal_1d)),
            signal_1d,
            alpha=0.3,
            where=mask
        )
        ax.set_title("XAI Explanation: Important ECG Regions")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)

        st.info(
            """
            Explanation:
            The highlighted regions show ECG waveform parts that strongly differ
            from the average signal. These regions usually include QRS complex
            and abnormal beat patterns.
            """
        )

        show_feature_explanation(signal_1d, result)


# -----------------------------
# TAB 2: ECG Image Upload
# -----------------------------
with tab2:
    uploaded_image = st.file_uploader(
        "Upload ECG Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image)

        st.image(
            img,
            caption="Uploaded ECG Image",
            use_container_width=True
        )

        extracted_signal = extract_signal_from_ecg_image(uploaded_image)

        st.subheader("Extracted ECG Signal from Image")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(extracted_signal)
        ax.set_title("Signal Extracted from ECG Image")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)

        signal_input = extracted_signal.reshape(1, 200, 1)

        prediction = model.predict(signal_input)[0]

        predicted_class = np.argmax(prediction)
        result = class_names[predicted_class]
        confidence = prediction[predicted_class] * 100
        risk = get_risk_level(result)

        if result == "Normal":
            st.success(f"Prediction from Image: {result}")
            st.success(f"Risk Level: {risk}")
        elif result == "PVC":
            st.warning(f"Prediction from Image: {result}")
            st.warning(f"Risk Level: {risk}")
        else:
            st.error(f"Prediction from Image: {result}")
            st.error(f"Risk Level: {risk}")

        st.write(f"Confidence: {confidence:.2f}%")

        show_feature_explanation(extracted_signal, result)

        st.warning(
            """
            Note:
            ECG image-to-signal extraction is experimental.
            For accurate prediction, use the .npy ECG signal upload.
            """
        )