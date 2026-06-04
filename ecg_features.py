import numpy as np
import neurokit2 as nk


def calculate_impact(value, normal_min, normal_max, max_impact):
    """
    Rule-based impact calculation.
    If value is inside normal range -> low impact.
    If outside normal range -> higher impact.
    """
    if value is None:
        return 0

    if normal_min <= value <= normal_max:
        return 5

    if value < normal_min:
        deviation = normal_min - value
    else:
        deviation = value - normal_max

    impact = min(max_impact, int((deviation / normal_max) * 100))
    return max(impact, 10)


def extract_ecg_features(ecg_signal, sampling_rate=360):
    """
    Extract Heart Rate, RR Interval, and QRS Width from ECG signal.
    """

    ecg_signal = np.array(ecg_signal).flatten()

    # Clean ECG
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

    # Detect R peaks
    signals, info = nk.ecg_process(cleaned, sampling_rate=sampling_rate)

    r_peaks = info.get("ECG_R_Peaks", [])

    # Heart Rate
    if "ECG_Rate" in signals.columns:
        heart_rate = round(float(np.nanmean(signals["ECG_Rate"])), 2)
    else:
        heart_rate = None

    # RR Interval in ms
    if len(r_peaks) > 1:
        rr_intervals_ms = np.diff(r_peaks) / sampling_rate * 1000
        rr_interval = round(float(np.nanmean(rr_intervals_ms)), 2)
        rr_variability = round(float(np.std(rr_intervals_ms)), 2)
    else:
        rr_interval = None
        rr_variability = None

    # QRS Width using ECG delineation
    qrs_width = None

    try:
        _, waves = nk.ecg_delineate(
            cleaned,
            r_peaks,
            sampling_rate=sampling_rate,
            method="dwt"
        )

        qrs_onsets = waves.get("ECG_R_Onsets", [])
        qrs_offsets = waves.get("ECG_R_Offsets", [])

        qrs_widths = []

        for onset, offset in zip(qrs_onsets, qrs_offsets):
            if not np.isnan(onset) and not np.isnan(offset):
                width_ms = (offset - onset) / sampling_rate * 1000
                qrs_widths.append(width_ms)

        if len(qrs_widths) > 0:
            qrs_width = round(float(np.nanmean(qrs_widths)), 2)

    except Exception:
        qrs_width = None

    # Rule-based impacts
    heart_rate_impact = calculate_impact(
        heart_rate,
        normal_min=60,
        normal_max=100,
        max_impact=30
    )

    qrs_width_impact = calculate_impact(
        qrs_width,
        normal_min=70,
        normal_max=120,
        max_impact=40
    )

    rr_impact = calculate_impact(
        rr_variability,
        normal_min=0,
        normal_max=120,
        max_impact=20
    )

    features = {
        "heart_rate": heart_rate,
        "heart_rate_impact": heart_rate_impact,

        "qrs_width": qrs_width,
        "qrs_width_impact": qrs_width_impact,

        "rr_interval": rr_interval,
        "rr_variability": rr_variability,
        "rr_impact": rr_impact
    }

    return features