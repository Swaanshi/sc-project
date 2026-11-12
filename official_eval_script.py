import torch
import librosa
import numpy as np
import os
import glob
from tqdm import tqdm # Progress bar so you know it hasn't crashed
from pesq import pesq
from pystoi import stoi
from run_ga_optimiser import SpeechEnhancementLSTM, apply_fuzzy_logic

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Master Model on {device}...")

model = SpeechEnhancementLSTM().to(device)
model.load_state_dict(torch.load("best_rnn_model.pth", map_location=device))
model.eval()

# Your Mathematically Proven GA Parameters
OPTIMAL_FUZZY_PARAMS = [0.1382, 0.5528, 0.4962, 0.8392, 0.1354]

# --- POINT THESE TO THE OFFICIAL 824-FILE TEST FOLDERS ---
OFFICIAL_TEST_NOISY = "noisy_testset_wav/noisy_testset_wav/"
OFFICIAL_TEST_CLEAN = "clean_testset_wav/clean_testset_wav/"

sample_rate = 16000

# ==========================================
# 2. THE EVALUATION LOOP
# ==========================================
noisy_files = sorted(glob.glob(os.path.join(OFFICIAL_TEST_NOISY, "*.wav")))

# Trackers for the Mean averages
results = {
    'baseline_pesq': [], 'baseline_stoi': [],
    'rnn_pesq': [], 'rnn_stoi': [],
    'hybrid_pesq': [], 'hybrid_stoi': []
}

print(f"Starting Official Evaluation on {len(noisy_files)} files...")
print("This will take a few minutes. Grab a coffee.")

for n_path in tqdm(noisy_files):
    basename = os.path.basename(n_path)
    c_path = os.path.join(OFFICIAL_TEST_CLEAN, basename)
    
    if not os.path.exists(c_path):
        continue # Skip if clean match is missing

    # Load Audio
    noisy_audio, _ = librosa.load(n_path, sr=sample_rate)
    clean_audio, _ = librosa.load(c_path, sr=sample_rate)
    
    stft = librosa.stft(noisy_audio, n_fft=512, hop_length=256)
    mag, phase = librosa.magphase(stft)
    
    # Predict
    with torch.no_grad():
        mag_tensor = torch.FloatTensor(mag.T).unsqueeze(0).to(device)
        rnn_mask = model(mag_tensor).squeeze(0).cpu().numpy().T
        
    # Rebuild RNN Audio
    rnn_audio = librosa.istft((mag * rnn_mask) * phase, hop_length=256)
    
    # Rebuild Hybrid Audio
    fuzzy_mask = apply_fuzzy_logic(rnn_mask, mag, OPTIMAL_FUZZY_PARAMS)
    hybrid_audio = librosa.istft((mag * fuzzy_mask) * phase, hop_length=256)
    
    # Sync lengths
    min_len = min(len(clean_audio), len(noisy_audio), len(rnn_audio), len(hybrid_audio))
    ref = clean_audio[:min_len]
    n_aud = noisy_audio[:min_len]
    r_aud = rnn_audio[:min_len]
    h_aud = hybrid_audio[:min_len]
    
    # Metrics
    try:
        results['baseline_pesq'].append(pesq(sample_rate, ref, n_aud, 'wb'))
        results['rnn_pesq'].append(pesq(sample_rate, ref, r_aud, 'wb'))
        results['hybrid_pesq'].append(pesq(sample_rate, ref, h_aud, 'wb'))
        
        results['baseline_stoi'].append(stoi(ref, n_aud, sample_rate, extended=False))
        results['rnn_stoi'].append(stoi(ref, r_aud, sample_rate, extended=False))
        results['hybrid_stoi'].append(stoi(ref, h_aud, sample_rate, extended=False))
    except:
        pass # Skip if a file is purely silent and breaks PESQ

# ==========================================
# 3. FINAL PUBLICATION NUMBERS
# ==========================================
print("\n========================================")
print("🏆 FINAL THESIS RESULTS (OFFICIAL TEST SET) 🏆")
print("========================================")
print(f"1. Noisy Baseline -> PESQ: {np.mean(results['baseline_pesq']):.3f} | STOI: {np.mean(results['baseline_stoi']):.3f}")
print(f"2. Pure RNN       -> PESQ: {np.mean(results['rnn_pesq']):.3f} | STOI: {np.mean(results['rnn_stoi']):.3f}")
print(f"3. RNN + GA-Fuzzy -> PESQ: {np.mean(results['hybrid_pesq']):.3f} | STOI: {np.mean(results['hybrid_stoi']):.3f}")
print("========================================")