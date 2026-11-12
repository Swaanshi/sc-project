import torch
import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from run_ga_optimiser import SpeechEnhancementLSTM, apply_fuzzy_logic # Imports your classes
from pystoi import stoi

# ---------------------------------------------------------
# 1. SETUP AND LOAD
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Master Model on {device}...")

# Load the trained LSTM
model = SpeechEnhancementLSTM().to(device)
model.load_state_dict(torch.load("best_rnn_model.pth", map_location=device))
model.eval()

# Your Mathematically Proven GA Parameters!
OPTIMAL_FUZZY_PARAMS = [0.1382, 0.5528, 0.4962, 0.8392, 0.1354]

# --- PICK AN UNSEEN TEST FILE HERE ---
# Make sure you have the noisy and clean versions
test_noisy_file = "micro_dataset/noisy/p226_016.wav" # Change this!
test_clean_file = "micro_dataset/clean/p226_016.wav" # Change this!

sample_rate = 16000

# ---------------------------------------------------------
# 2. PROCESS THE AUDIO
# ---------------------------------------------------------
print(f"Processing File: {test_noisy_file}")
noisy_audio, _ = librosa.load(test_noisy_file, sr=sample_rate)
clean_audio, _ = librosa.load(test_clean_file, sr=sample_rate)

stft = librosa.stft(noisy_audio, n_fft=512, hop_length=256)
mag, phase = librosa.magphase(stft)

# --- A: The RNN Only (Industry Standard) ---
with torch.no_grad():
    mag_tensor = torch.FloatTensor(mag.T).unsqueeze(0).to(device)
    rnn_mask = model(mag_tensor).squeeze(0).cpu().numpy().T

rnn_clean_mag = mag * rnn_mask
rnn_audio = librosa.istft(rnn_clean_mag * phase, hop_length=256)

# --- B: The Neuro-Fuzzy (Your Novelty) ---
fuzzy_mask = apply_fuzzy_logic(rnn_mask, mag, OPTIMAL_FUZZY_PARAMS)
fuzzy_clean_mag = mag * fuzzy_mask
fuzzy_audio = librosa.istft(fuzzy_clean_mag * phase, hop_length=256)

# ---------------------------------------------------------
# 3. SAVE AUDIO & CALCULATE SCORES
# ---------------------------------------------------------
# Save the files so you can play them in your presentation
sf.write("1_thesis_noisy_baseline.wav", noisy_audio, sample_rate)
sf.write("2_thesis_rnn_only.wav", rnn_audio, sample_rate)
sf.write("3_thesis_rnn_fuzzy_hybrid.wav", fuzzy_audio, sample_rate)

# Calculate PESQ Scores (Truncate to match lengths exactly)
min_len = min(len(clean_audio), len(noisy_audio), len(rnn_audio), len(fuzzy_audio))
clean_ref = clean_audio[:min_len]

# Add this import at the top
from pystoi import stoi

# ... (down in the calculation section) ...

# Calculate PESQ
score_noisy_pesq = pesq(sample_rate, clean_ref, noisy_audio[:min_len], 'wb')
score_rnn_pesq = pesq(sample_rate, clean_ref, rnn_audio[:min_len], 'wb')
score_fuzzy_pesq = pesq(sample_rate, clean_ref, fuzzy_audio[:min_len], 'wb')

# Calculate STOI
score_noisy_stoi = stoi(clean_ref, noisy_audio[:min_len], sample_rate, extended=False)
score_rnn_stoi = stoi(clean_ref, rnn_audio[:min_len], sample_rate, extended=False)
score_fuzzy_stoi = stoi(clean_ref, fuzzy_audio[:min_len], sample_rate, extended=False)

print("\n========================================")
print("🎯 THESIS ABLATION STUDY RESULTS 🎯")
print("========================================")
print(f"1. Noisy Baseline -> PESQ: {score_noisy_pesq:.4f} | STOI: {score_noisy_stoi:.4f}")
print(f"2. Pure RNN       -> PESQ: {score_rnn_pesq:.4f} | STOI: {score_rnn_stoi:.4f}")
print(f"3. RNN + GA-Fuzzy -> PESQ: {score_fuzzy_pesq:.4f} | STOI: {score_fuzzy_stoi:.4f}")
print("========================================")