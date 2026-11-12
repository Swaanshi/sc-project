import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import glob
import random

# ==========================================
# 1. THE ARCHITECTURE
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
# 2. THE AUTOMATED DATA SPLITTER
# ==========================================
def prepare_dataset_splits(master_noisy_dir, master_clean_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Pairs the files, shuffles them, and splits them into Train, Val, and Test lists.
    """
    print("Scanning master folders and verifying file pairs...")
    noisy_candidates = sorted(glob.glob(os.path.join(master_noisy_dir, "*.wav")))
    
    paired_noisy = []
    paired_clean = []
    
    # Strict Pairing logic
    for n_path in noisy_candidates:
        basename = os.path.basename(n_path)
        c_path = os.path.join(master_clean_dir, basename)
        
        if os.path.exists(c_path):
            paired_noisy.append(n_path)
            paired_clean.append(c_path)
            
    total_files = len(paired_noisy)
    if total_files == 0:
        raise ValueError("Zero matching files found! Check your master folder paths.")
        
    print(f"Successfully paired {total_files} audio files. Shuffling and splitting...")
    
    # Zip together, shuffle, and unzip to maintain perfect pairs
    combined = list(zip(paired_noisy, paired_clean))
    random.seed(42) # Set seed so the split is exactly the same every time you run the script
    random.shuffle(combined)
    paired_noisy, paired_clean = zip(*combined)
    
    # Calculate split indices
    train_end = int(total_files * train_ratio)
    val_end = int(total_files * (train_ratio + val_ratio))
    
    splits = {
        'train_noisy': paired_noisy[:train_end],
        'train_clean': paired_clean[:train_end],
        'val_noisy': paired_noisy[train_end:val_end],
        'val_clean': paired_clean[train_end:val_end],
        'test_noisy': paired_noisy[val_end:],
        'test_clean': paired_clean[val_end:]
    }
    
    print(f"Data Split Complete:")
    print(f" - Training Files:   {len(splits['train_noisy'])}")
    print(f" - Validation Files: {len(splits['val_noisy'])}")
    print(f" - Test Files:       {len(splits['test_noisy'])}\n")
    
    return splits

# ==========================================
# 3. THE DATA LOADER (Now accepts lists instead of folders)
# ==========================================
class VoiceBankDataset(Dataset):
    def __init__(self, noisy_files_list, clean_files_list, chunk_duration=2.0, sr=16000):
        self.chunk_samples = int(chunk_duration * sr)
        self.sr = sr
        self.noisy_files = noisy_files_list
        self.clean_files = clean_files_list

    def __len__(self):
        return len(self.noisy_files)
        
    def __getitem__(self, idx):
        noisy_audio, _ = librosa.load(self.noisy_files[idx], sr=self.sr)
        clean_audio, _ = librosa.load(self.clean_files[idx], sr=self.sr)
        
        # Slicing/Padding to 2 seconds for batch processing
        if len(noisy_audio) > self.chunk_samples:
            start = np.random.randint(0, len(noisy_audio) - self.chunk_samples)
            noisy_audio = noisy_audio[start:start + self.chunk_samples]
            clean_audio = clean_audio[start:start + self.chunk_samples]
        else:
            noisy_audio = np.pad(noisy_audio, (0, self.chunk_samples - len(noisy_audio)))
            clean_audio = np.pad(clean_audio, (0, self.chunk_samples - len(clean_audio)))
            
        noisy_stft = librosa.stft(noisy_audio, n_fft=512, hop_length=256)
        clean_stft = librosa.stft(clean_audio, n_fft=512, hop_length=256)
        
        noisy_mag = np.abs(noisy_stft)
        clean_mag = np.abs(clean_stft)
        
        # Ideal Ratio Mask (IRM) calculation
        target_mask = clean_mag / (noisy_mag + 1e-8)
        target_mask = np.clip(target_mask, 0.0, 1.0)
        
        return torch.FloatTensor(noisy_mag.T), torch.FloatTensor(target_mask.T)

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
def train_rnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Initialization ---")
    print(f"Hardware Detected: {device}")
    
    # ---------------------------------------------------------
    # UPDATE THESE TWO PATHS TO YOUR ACTUAL MASTER FOLDERS
    # ---------------------------------------------------------
    MASTER_NOISY = "noisy_trainset_28spk_wav/noisy_trainset_28spk_wav/"
    MASTER_CLEAN = "clean_trainset_28spk_wav/clean_trainset_28spk_wav/"
    
    # Auto-Split the Data
    splits = prepare_dataset_splits(MASTER_NOISY, MASTER_CLEAN)
    
    # Load into PyTorch DataLoaders
    train_loader = DataLoader(VoiceBankDataset(splits['train_noisy'], splits['train_clean']), batch_size=16, shuffle=True)
    val_loader   = DataLoader(VoiceBankDataset(splits['val_noisy'],   splits['val_clean']),   batch_size=16, shuffle=False)
    test_loader  = DataLoader(VoiceBankDataset(splits['test_noisy'],  splits['test_clean']),  batch_size=16, shuffle=False)
    
    # Initialize Model & Optimizer
    model = SpeechEnhancementLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    best_val_loss = float('inf') 
    
    print("Starting Training Phase...")
    for epoch in range(num_epochs):
        
        # --- 1. TRAINING LOOP ---
        model.train()
        running_train_loss = 0.0
        for batch_idx, (noisy_mag, target_mask) in enumerate(train_loader):
            noisy_mag, target_mask = noisy_mag.to(device), target_mask.to(device)
            
            optimizer.zero_grad()
            predicted_mask = model(noisy_mag)
            loss = criterion(predicted_mask, target_mask)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = running_train_loss / len(train_loader)
        
        # --- 2. VALIDATION LOOP ---
        model.eval() 
        running_val_loss = 0.0
        with torch.no_grad(): 
            for val_noisy, val_clean in val_loader:
                val_noisy, val_clean = val_noisy.to(device), val_clean.to(device)
                val_pred = model(val_noisy)
                v_loss = criterion(val_pred, val_clean)
                running_val_loss += v_loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss:   {avg_val_loss:.4f}")
        
        # --- 3. THE "BEST MODEL" CHECKPOINT ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_rnn_model.pth")
            print("--> Validation loss decreased! Saved new best model.\n")
        else:
            print("--> Validation loss did not improve.\n")

    # --- 4. THE FINAL TEST EXAM ---
    print("=========================================")
    print("Training Complete. Loading Best Model for Final Testing...")
    
    model.load_state_dict(torch.load("best_rnn_model.pth"))
    model.eval()
    running_test_loss = 0.0
    
    with torch.no_grad():
        for test_noisy, test_clean in test_loader:
            test_noisy, test_clean = test_noisy.to(device), test_clean.to(device)
            test_pred = model(test_noisy)
            t_loss = criterion(test_pred, test_clean)
            running_test_loss += t_loss.item()
            
    final_test_loss = running_test_loss / len(test_loader)
    print(f"*** FINAL UNSEEN TEST LOSS (MSE): {final_test_loss:.4f} ***")
    print("=========================================")

if __name__ == "__main__":
    train_rnn()