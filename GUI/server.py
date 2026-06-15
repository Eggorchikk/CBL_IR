"""
IR Textile Classifier — local backend
======================================
Run with:  python server.py
Then open: index.html in your browser (or visit http://localhost:5000)

Expects model files:
  models/pure_textile_model.joblib   — sklearn Pipeline (PCA + SVM)
  models/composition_cnn.pt          — CompositionCNN checkpoint dict

CSV format (two columns, no header, comma-separated):
  wavenumber,intensity
  3999.0,0.123
  3998.0,0.118
  ...
"""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

PURE_MODEL_PATH  = "models/pure_textile_model.joblib"
MIXED_MODEL_PATH = "models/composition_cnn.pt"

# Wavenumber grid — must match what the models were trained on
WN_COMMON = np.arange(3999.0, 674.0, -1)   # 3326 points


# ════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (must match training code exactly)
# ════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class CompositionCNN(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            ResBlock1D(32, kernel_size=9),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            ResBlock1D(64, kernel_size=5),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            ResBlock1D(128, kernel_size=3),
            nn.AdaptiveAvgPool1d(32),
            nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (done once at startup)
# ════════════════════════════════════════════════════════════════════

def load_pure_model(path):
    """
    Loads the sklearn PCA+SVM pipeline saved with joblib.

    Save it from your notebook with:
        import joblib
        pca_svm.fit(X, labels)          # fit on full training set first
        joblib.dump(pca_svm, "models/pure_textile_model.joblib")
    """
    pipeline = joblib.load(path)
    return pipeline


def load_mixed_model(path):
    """
    Loads the CompositionCNN from the checkpoint dict saved with torch.save().

    The checkpoint must contain: model_state_dict, classes, n_wn, n_classes.
    This matches exactly what your notebook saves:
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes':          classes,
            'n_wn':             n_wn,
            'n_classes':        n_classes,
        }, "composition_cnn.pt")
    """
    ckpt     = torch.load(path, map_location="cpu")
    classes  = ckpt["classes"]
    n_wn     = ckpt["n_wn"]
    n_classes = ckpt["n_classes"]

    model = CompositionCNN(n_wn, n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes


print("Loading models…")

try:
    pure_pipeline = load_pure_model(PURE_MODEL_PATH)
    # Read class list from the fitted SVM
    pure_classes  = list(pure_pipeline.classes_)
    print(f"  ✓ Pure model loaded  — classes: {pure_classes}")
except Exception as e:
    pure_pipeline = None
    pure_classes  = []
    print(f"  ✗ Pure model NOT loaded: {e}")

try:
    mixed_model, mixed_classes = load_mixed_model(MIXED_MODEL_PATH)
    mixed_classes = list(mixed_classes)
    print(f"  ✓ Mixed model loaded — classes: {mixed_classes}")
except Exception as e:
    mixed_model   = None
    mixed_classes = []
    print(f"  ✗ Mixed model NOT loaded: {e}")


# ════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (mirrors the notebook exactly)
# ════════════════════════════════════════════════════════════════════

def als_baseline(y, D, L, p=0.01, niter=10):
    """Asymmetric Least Squares baseline correction (from notebook)."""
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def preprocess_spectrum(file_bytes):
    """
    Full preprocessing pipeline applied to a single raw CSV spectrum.

    Steps (identical to notebook):
      1. Load CSV (two columns: wavenumber, intensity)
      2. Interpolate onto WN_COMMON grid (cubic, with NaN fill)
      3. ALS baseline correction  (lam=1e5, p=0.01, niter=10)
      4. SNV normalisation         (z-score: subtract mean, divide by std)
      5. Savitzky-Golay smoothing  (window=9, polyorder=3)

    Returns:
      np.ndarray of shape (3326,), dtype float32
    """
    # ── 1. Parse CSV ────────────────────────────────────────────
    # Tries comma-separated first; falls back to semicolon (KARLILE format)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None)
        if df.shape[1] < 2:
            raise ValueError("Need at least 2 columns")
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=";")

    df = df.replace("#NaN", np.nan)
    wn_src = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    ab_src = pd.to_numeric(df.iloc[:, 1], errors="coerce").values

    mask   = ~np.isnan(wn_src) & ~np.isnan(ab_src)
    wn_src = wn_src[mask]
    ab_src = ab_src[mask]

    if len(wn_src) < 4:
        raise ValueError("CSV has fewer than 4 valid data points.")

    # ── 2. Interpolate onto common grid ─────────────────────────
    sort_idx = np.argsort(wn_src)
    wn_src   = wn_src[sort_idx]
    ab_src   = ab_src[sort_idx]

    f_interp    = interp1d(wn_src, ab_src, kind="cubic",
                           bounds_error=False, fill_value=np.nan)
    interpolated = f_interp(WN_COMMON)

    # Fill any remaining NaNs by linear interpolation at boundaries
    nans = np.isnan(interpolated)
    if nans.all():
        raise ValueError(
            "Spectrum wavenumber range does not overlap the required grid "
            f"({WN_COMMON[-1]:.0f}–{WN_COMMON[0]:.0f} cm⁻¹)."
        )
    if nans.any():
        non_nan = ~nans
        interpolated[nans] = np.interp(
            np.where(nans)[0],
            np.where(non_nan)[0],
            interpolated[non_nan],
        )

    # ── 3. ALS baseline correction ───────────────────────────────
    L = len(interpolated)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = 1e5 * D.T @ D
    baseline     = als_baseline(interpolated, D, L)
    corrected    = interpolated - baseline

    # ── 4. SNV normalisation ─────────────────────────────────────
    mu    = corrected.mean()
    sigma = corrected.std()
    if sigma == 0:
        raise ValueError("Spectrum has zero variance after baseline correction.")
    normalised = (corrected - mu) / sigma

    # ── 5. Savitzky-Golay smoothing ──────────────────────────────
    smoothed = savgol_filter(normalised, window_length=9, polyorder=3)

    return smoothed.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  CLASSIFICATION ENDPOINT
# ════════════════════════════════════════════════════════════════════

@app.route("/classify", methods=["POST"])
def classify():
    # ── Validate request ─────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    model_type = request.form.get("model", "pure")   # "pure" or "mixed"
    file_bytes = request.files["file"].read()

    # ── Preprocess ───────────────────────────────────────────────
    try:
        spectrum = preprocess_spectrum(file_bytes)   # shape (3326,)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    # ════════════════════════════════════════════════════════════
    #  PURE TEXTILE  — PCA + SVM pipeline
    # ════════════════════════════════════════════════════════════
    if model_type == "pure":
        if pure_pipeline is None:
            return jsonify({"error": f"Pure model not loaded. Check {PURE_MODEL_PATH}."}), 500

        try:
            X = spectrum.reshape(1, -1)   # (1, 3326)

            # Predicted class label
            prediction = pure_pipeline.predict(X)[0]

            # Decision function scores (one per class) → convert to pseudo-probabilities
            # SVC.decision_function returns shape (1, n_classes) for multi-class
            dec_scores = pure_pipeline.decision_function(X)[0]   # (n_classes,)

            # Softmax over decision scores for display (not true probabilities,
            # but a calibrated-enough signal for the logit bars)
            exp_scores = np.exp(dec_scores - dec_scores.max())
            probs      = (exp_scores / exp_scores.sum()).tolist()
            logits     = dec_scores.tolist()
            classes    = pure_classes

        except Exception as e:
            return jsonify({"error": f"Pure model inference failed: {e}"}), 500

        confidence = probs[classes.index(prediction)]

    # ════════════════════════════════════════════════════════════
    #  MIXED TEXTILE  — CompositionCNN
    # ════════════════════════════════════════════════════════════
    else:
        if mixed_model is None:
            return jsonify({"error": f"Mixed model not loaded. Check {MIXED_MODEL_PATH}."}), 500

        try:
            # Shape: (1, 1, 3326) — batch=1, channels=1, length=3326
            x_tensor = torch.tensor(spectrum[None, None, :], dtype=torch.float32)

            with torch.no_grad():
                output = mixed_model(x_tensor)   # (1, n_classes) — already softmaxed

            probs  = output.squeeze().tolist()    # fractional compositions 0–1
            classes = mixed_classes

            # For the mixed model the "logits" shown are the raw pre-softmax
            # activations — we back-compute them for display purposes
            # (log of softmax output, shifted for readability)
            logits = np.log(np.array(probs) + 1e-8).tolist()

            # Top 2 classes by fraction
            top2_idx   = np.argsort(probs)[::-1][:2].tolist()
            prediction = classes[top2_idx[0]]
            confidence = probs[top2_idx[0]]
            top2       = [{"label": classes[i], "prob": probs[i]} for i in top2_idx]

        except Exception as e:
            return jsonify({"error": f"Mixed model inference failed: {e}"}), 500

    # ── Build response ───────────────────────────────────────────
    response = {
        "prediction": prediction,     # string: top class label
        "confidence": confidence,     # float 0-1
        "model":      model_type,     # "pure" or "mixed"
        "classes":    classes,        # list of class label strings
        "probs":      probs,          # list of floats (softmax or composition fractions)
        "logits":     logits,         # list of floats (decision scores or log-probs)
        "top2":       top2 if model_type == "mixed" else None,
    }

    return jsonify(response)


# ════════════════════════════════════════════════════════════════════
#  SERVE  index.html  at  http://localhost:5000
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    print("\n── IR Textile Classifier ───────────────────────────────")
    print("   Server: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(host="localhost", port=5000, debug=False)
