"""Studio deployment input contracts (not changes to the calibrated model)."""

LIMITED_MODE = "experimental-96k-bandlimited-v1"
LIMITED_NOTICE = ("Experimental 96 kHz mode: detections are limited to 4–48 kHz. "
                  "Accuracy at this recording rate has not been validated.")

def rate_supported(rate):
    return rate == 96_000 or rate >= 192_000

def compatibility_metadata(rate):
    if rate != 96_000:
        return {}
    return {"input_mode": LIMITED_MODE, "source_sample_rate_hz": rate,
            "model_input_rate_hz": 192_000, "resampling": "per-window scipy.signal.resample_poly 2:1 upsampling",
            "detection_band_hz": [4_000, 48_000], "accuracy_validated_at_source_rate": False}
