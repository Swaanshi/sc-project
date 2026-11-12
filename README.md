# Speech Enhancement using RNN and Neuro-Fuzzy Hybrid Approach

## Project Overview

This project presents a speech enhancement system designed to improve the quality and intelligibility of noisy audio signals. The system combines a deep learning model with a rule-based post-processing technique to achieve better performance than traditional methods.

The core of the system is a Bidirectional Long Short-Term Memory (BiLSTM) network that learns to estimate a mask over the noisy speech signal. This mask is applied to the magnitude spectrogram to suppress noise components. To further refine the output, a Neuro-Fuzzy system is introduced, whose parameters are optimized using a Genetic Algorithm (GA).

This hybrid approach allows the system to benefit from both data-driven learning and interpretable rule-based corrections, resulting in improved perceptual quality.

---

## Project Structure

The project is organized into multiple scripts, each responsible for a specific stage of the pipeline:

2023UIT3007_2023UIT3070_SpeechEnhancement/

- train_lstm_v2.py  
- run_ga_optimiser.py  
- generate_thesis_result.py  
- official_eval_script.py  

- best_rnn_model.pth  
- rnn_checkpoint_epoch_1.pth  

- micro_dataset/  
  - noisy/  
  - clean/  

- output_audio/  


---

## Software and Libraries Required

This project is implemented in Python and requires Python version 3.8 or above.

Install the required libraries using:

pip install torch librosa numpy soundfile pesq pystoi tqdm pygad

The libraries serve the following purposes:

PyTorch is used to build and train the deep learning model.  
Librosa is used for audio processing and spectrogram computation.  
NumPy is used for numerical operations.  
SoundFile is used for saving processed audio outputs.  
PESQ and STOI are used for evaluating speech quality and intelligibility.  
PyGAD is used for optimizing fuzzy logic parameters.  
tqdm is used to display progress during long-running operations.

---

## How to Run the Code

### 1. Training the RNN Model

python train_lstm_v2.py

This script trains the model and saves the best model as:

best_rnn_model.pth

---

### 2. Optimizing Fuzzy Parameters

python run_ga_optimiser.py

This script optimizes fuzzy parameters using a Genetic Algorithm.

---

### 3. Generating Sample Results

python generate_thesis_result.py

This generates enhanced audio outputs and prints evaluation metrics.

---

### 4. Running Full Evaluation

python official_eval_script.py

This evaluates the model on the full dataset and reports average scores.

---

## Input Format

The input consists of .wav audio files sampled at 16 kHz.

The dataset must be organized such that each noisy file has a corresponding clean file with the same name.

Example:

noisy/file.wav  
clean/file.wav  

---

## Output Format

The system generates enhanced audio files:

1_thesis_noisy_baseline.wav  
2_thesis_rnn_only.wav  
3_thesis_rnn_fuzzy_hybrid.wav  

It also prints evaluation metrics:

PESQ and STOI scores for each method.

---

## Methodology

The system operates in three stages.

First, a Bidirectional LSTM predicts a mask over the noisy speech spectrogram.

Second, a Neuro-Fuzzy system refines uncertain regions using rule-based adjustments.

Finally, a Genetic Algorithm optimizes the fuzzy parameters to maximize perceptual quality.

---

## Important Notes

Ensure dataset paths are correctly set before running the scripts.  
GPU is recommended for faster training but not mandatory.  
PESQ may fail for silent audio segments, which is handled in the code.  
Proper pairing of noisy and clean files is essential.

---
