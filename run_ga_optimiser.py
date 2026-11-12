import torch
import torch.nn as nn
import librosa
import numpy as np
import pygad
import os
import glob
from pesq import pesq

# ==========================================
# 1. THE ARCHITECTURE (Must match Phase 1 exactly)
# ==========================================
class SpeechEnhancementLSTM(nn.Module):
    def __init__(self, input_dim=257, hidden_dim=256, num_layers=2):
        super(SpeechEnhancementLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mask = self.fc(lstm_out)
        return self.sigmoid(mask)

# ==========================================
# 2. THE NEURO-FUZZY POST-PROCESSOR
# ==========================================
def apply_fuzzy_logic(rnn_mask, magnitude, fuzzy_params):
    """
    Applies vector-optimized Fuzzy boundaries to the RNN output.
    fuzzy_params = [u_lower, u_upper, e_thresh, s_boost, n_drop]
    """
    u_lower, u_upper, e_thresh, s_boost, n_drop = fuzzy_params
    fuzzy_mask = np.copy(rnn_mask)
    
    # Normalize energy to 0-1 scale for the Fuzzy Threshold
    max_energy = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    norm_energy = magnitude / max_energy
    
    uncertain_zones = (rnn_mask > u_lower) & (rnn_mask < u_upper)
    
    # Rule A: Uncertain + Loud = Rescue the Speech Consonants
    speech_rescue = uncertain_zones & (norm_energy > e_thresh)
    fuzzy_mask[speech_rescue] = s_boost
    
    # Rule B: Uncertain + Quiet = Kill the Residual Noise
    noise_kill = uncertain_zones & (norm_energy <= e_thresh)
    fuzzy_mask[noise_kill] = n_drop
    
    return np.clip(fuzzy_mask, 0.0, 1.0)

# ==========================================
# 3. LOAD THE RNN AND PREPARE MICRO-DATASET
# ==========================================
print("--- Initializing Phase 2: Neuro-Fuzzy Optimization ---")

# 1. Setup hardware and load the trained Phase 1 Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading best_rnn_model.pth onto {device}...")

rnn_model = SpeechEnhancementLSTM().to(device)
rnn_model.load_state_dict(torch.load('best_rnn_model.pth', map_location=device))
rnn_model.eval() # STRICTLY Testing mode (no learning)

# 2. Setup Micro-Dataset paths
# Create a folder called 'micro_dataset' in your project directory
# Inside it, create 'noisy' and 'clean', and put your 5 diverse files in them.
MICRO_NOISY_DIR = "micro_dataset/noisy/"
MICRO_CLEAN_DIR = "micro_dataset/clean/"
sample_rate = 16000
validation_batch = []

print("Extracting Deep Learning Baselines for Micro-Dataset...")
noisy_files = sorted(glob.glob(os.path.join(MICRO_NOISY_DIR, "*.wav")))

for n_path in noisy_files:
    basename = os.path.basename(n_path)
    c_path = os.path.join(MICRO_CLEAN_DIR, basename)
    
    if os.path.exists(c_path):
        # Load audio
        n_audio, _ = librosa.load(n_path, sr=sample_rate)
        c_audio, _ = librosa.load(c_path, sr=sample_rate)
        
        # Extract Phase and Magnitude
        stft = librosa.stft(n_audio, n_fft=512, hop_length=256)
        mag, phase = librosa.magphase(stft)
        
        # --- THE DEEP LEARNING PREDICTION ---
        # The RNN only runs ONCE per file here, saving massive compute time for the GA.
        with torch.no_grad():
            mag_tensor = torch.FloatTensor(mag.T).unsqueeze(0).to(device)
            # Predict, move back to CPU, detach from graph, and transpose back
            base_rnn_mask = rnn_model(mag_tensor).squeeze(0).cpu().detach().numpy().T
            
        validation_batch.append({
            'filename': basename,
            'noisy_mag': mag,
            'phase': phase,
            'base_mask': base_rnn_mask,
            'clean_audio_ref': c_audio
        })

if len(validation_batch) == 0:
    raise ValueError("No files found in micro_dataset folders! Please add your 5 files.")
    
print(f"Successfully processed {len(validation_batch)} files. GA is ready.")

# ==========================================
# 4. THE FITNESS EVALUATOR (MEAN PESQ)
# ==========================================
def fitness_func(ga_instance, solution, solution_idx):
    batch_scores = []
    
    for data in validation_batch:
        # 1. Apply the GA's current guess for Fuzzy Rules
        refined_mask = apply_fuzzy_logic(data['base_mask'], data['noisy_mag'], solution)
        
        # 2. Rebuild Audio (Phase Intact!)
        clean_stft = (data['noisy_mag'] * refined_mask) * data['phase']
        clean_audio = librosa.istft(clean_stft, hop_length=256)
        
        # 3. Evaluate Perceptual Quality with PESQ
        ref_audio = data['clean_audio_ref']
        min_len = min(len(ref_audio), len(clean_audio))
        
        try:
            # PESQ Wideband mode (standard for 16kHz audio)
            score = pesq(sample_rate, ref_audio[:min_len], clean_audio[:min_len], 'wb')
        except:
            score = 1.0 # Heavy penalty if the mathematical structure collapses
            
        batch_scores.append(score)
        
    # The GA is rewarded for the MEAN perceptual score across all diverse noise types
    return float(np.mean(batch_scores))

# ==========================================
# 5. EXECUTE THE GENETIC ALGORITHM
# ==========================================
if __name__ == "__main__":
    print("\nStarting Genetic Evolution...")
    print("Optimizing Fuzzy Inference Boundaries against PESQ metric.")
    
    # The DNA boundaries for our 5 Fuzzy Genes
    gene_space = [
        {'low': 0.1, 'high': 0.45},  # uncertain_lower
        {'low': 0.55, 'high': 0.9},  # uncertain_upper
        {'low': 0.05, 'high': 0.5},  # energy_threshold
        {'low': 0.8, 'high': 1.0},   # speech_boost
        {'low': 0.0, 'high': 0.2}    # noise_drop
    ]

    ga_instance = pygad.GA(
        num_generations=25,     # 25 Generations is optimal for 5 parameters
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=15,         # 15 individuals per generation
        num_genes=5,
        gene_space=gene_space,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20
    )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    
    print("\n========================================")
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print("========================================")
    print(f"Optimal Fuzzy Boundaries Discovered:")
    print(f" - Uncertain Lower Bound:   {solution[0]:.4f}")
    print(f" - Uncertain Upper Bound:   {solution[1]:.4f}")
    print(f" - Acoustic Energy Trigger: {solution[2]:.4f}")
    print(f" - Speech Rescue Boost:     {solution[3]:.4f}")
    print(f" - Noise Kill Drop:         {solution[4]:.4f}")
    print("----------------------------------------")
    print(f"Maximized Mean PESQ Score: {solution_fitness:.4f}")
    print("========================================")