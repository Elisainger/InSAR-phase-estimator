# InSAR Denoising with CNNs (Unsupervised vs Supervised)

This project investigates **Convolutional Neural Network (CNN)** approaches for **Interferometric Synthetic Aperture Radar (InSAR) denoising**, with a focus on comparing:

- **Unsupervised autoencoder training**
- **Supervised training using simulated ground truth**

InSAR data provide valuable information about surface deformation, but are often corrupted by noise. This project explores how deep learning can be used to improve phase reconstruction quality.

---

# Project Structure

.
├── train_coh_unsupervised.py # Unsupervised training (noisy → noisy)
├── train_coh_supervised.py # Supervised training (noisy → clean)
├── compare.py # Compare input, reconstruction, ground truth
├── buildset_noisy.py # Simulation script
├── InSAR-simulator/ # External simulator repo
├── simtdset/
│ ├── noisy/ # Simulated noisy interferograms
│ ├── clean/ # Simulated clean interferograms
├── train/
│ ├── ifg_ae/ # Saved model weights
│ ├── ifg_patches.hdf5 # Training dataset


---

# Setup Instructions

## 1. Create and activate environment


conda create -n name_of_env python=3.10
conda activate name_of_env


## 2. Install dependencies


pip install numpy matplotlib tensorflow keras h5py scikit-image scikit-learn


---

# Step 1: Generate Simulated Data


In the project folder:


InSAR-simulator/


Generate data:


python buildset_noisy.py


This creates:


simtdset/
├── noisy/
├── clean/


Each file is paired:


noisy/10.npy ↔ clean/10.npy


---

# Step 2: Train Models

## Unsupervised Training


python train_coh_unsupervised.py


- Input: noisy interferograms  
- Target: noisy interferograms  
- Learns denoising via dimensionality reduction  

---

## Supervised Training


python train_coh_supervised.py


- Input: noisy interferograms  
- Target: clean interferograms  
- Learns direct noise removal  

---

## Output

Models are saved in:


train/ifg_ae/


Example:


weights.01.keras
weights.02.keras
...


---

# Step 3: Compare Results


python compare.py


This script:

- Loads a test interferogram  
- Runs the trained model  
- Plots:
  - Input (noisy phase)
  - Reconstructed phase
  - Clean ground truth  

It also prints:

- **MSE (Mean Squared Error)**
- **PHCE (Phase Cosine Error)**

---

# Evaluation Metrics

**MSE**
- Pixel-wise error  
- Lower is better  

**PHCE**
- Measures phase similarity  
- Closer to 1 is better  

---

# Method Overview

**Unsupervised Autoencoder**
- Trained only on noisy data  
- Learns compressed representation  
- Denoising is implicit  
- More applicable to real InSAR data since no clean ground truth is usually available

**Supervised CNN**
- Uses simulated clean data  
- Learns mapping: noisy → clean  
- Typically achieves stronger denoising but requires clean ground truth

---


# References

Mukherjee et al.  
*CNN-Based InSAR Denoising and Coherence Metric*, IEEE SENSORS, 2018.

---


# Summary

This project demonstrates:

- CNN-based InSAR denoising  
- Comparison of unsupervised vs supervised learning  
- Use of simulated data for evaluation  