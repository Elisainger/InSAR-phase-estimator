import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_utils import saturate_outlier
from skimage.transform import resize


def process_ifg(input_ifg):
    Z_processed = saturate_outlier(input_ifg)
    Z_real = np.expand_dims(Z_processed.real + 1, axis=-1)
    Z_imag = np.expand_dims(Z_processed.imag + 1, axis=-1)
    return np.concatenate((Z_real, Z_imag), axis=-1)


def build_ifg(input_pred):
    out_ifg_real = input_pred[:, :, :, 0]
    out_ifg_imag = input_pred[:, :, :, 1]
    out_ifg = out_ifg_real + 1j * out_ifg_imag
    out_ifg -= (1 + 1j)
    return out_ifg


def resize_pred(pred, out_height, out_width):
    out_res = np.zeros((1, out_height, out_width, 2), dtype=float)

    pred = pred.copy()
    pred[0, :, :, 0] = np.clip(pred[0, :, :, 0] - 1, a_min=-1.0, a_max=1.0)
    pred[0, :, :, 1] = np.clip(pred[0, :, :, 1] - 1, a_min=-1.0, a_max=1.0)

    out_res[0, :, :, 0] = resize(pred[0, :, :, 0], (out_height, out_width))
    out_res[0, :, :, 1] = resize(pred[0, :, :, 1], (out_height, out_width))

    out_res[0, :, :, 0] += 1
    out_res[0, :, :, 1] += 1
    return out_res


# --- paths ---
model_path = r"train\unsupervised\ifg_ae\weights.10.keras"
noisy_path = r"simtdset\noisy\70.npy"
clean_path = r"simtdset\clean\70.npy"

# --- load model ---
model = load_model(model_path, compile=False)

# --- load data ---
x_noisy = np.load(noisy_path)
x_clean = np.load(clean_path)

# --- convert to phase-only ---
noisy_phase = np.angle(x_noisy)
clean_phase = np.angle(x_clean)

x_noisy_complex = np.cos(noisy_phase) + 1j * np.sin(noisy_phase)

# --- preprocess ---
x_in = process_ifg(x_noisy_complex)

# --- predict ---
pred = model.predict(np.expand_dims(x_in, axis=0), verbose=0)

# --- resize ---
h, w = x_noisy.shape
pred_resized = resize_pred(pred, h, w)

# --- reconstruct ---
x_rec = np.squeeze(build_ifg(pred_resized))
recon_phase = np.angle(x_rec)

# --- MSE (simple, not wrapped) ---
mse = np.mean((recon_phase - clean_phase) ** 2)
print(f"MSE (reconstruction vs clean): {mse:.6f}")

phase_error = np.angle(np.exp(1j * (recon_phase - clean_phase)))
phce = np.mean(np.cos(np.abs(phase_error)))

print(f"PHCE (reconstruction vs clean): {phce:.6f}")

# --- plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(noisy_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[0].set_title("Input (Noisy Phase)")
axes[0].axis("off")

axes[1].imshow(recon_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[1].set_title("Reconstructed Phase")
axes[1].axis("off")

axes[2].imshow(clean_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[2].set_title("Clean Ground Truth")
axes[2].axis("off")

plt.savefig("comparison.png", dpi=200, bbox_inches="tight")
plt.show()