# ECG Classification using 1D CNN + Progressive Data Dropout(PDD)

A deep learning pipeline for ECG signal classification using a 1D Convolutional Neural Network and Progressive Data Dropout (PDD), where the model is trained in stages from easy → medium → hard samples.


PTB-XL Dataset (Official Download — PhysioNet)
Main dataset page:

https://physionet.org/content/ptb-xl/1.0.3/

Direct download links:

Full PTB-XL Dataset (1 GB):
https://physionet.org/files/ptb-xl/1.0.3/ptb-xl-a.tar.gz

WFDB Records (raw ECG signals):
https://physionet.org/files/ptb-xl/1.0.3/records500.tar.gz

https://physionet.org/files/ptb-xl/1.0.3/records100.tar.gz

Metadata CSV files:
https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv

https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv



# Project Overview

This project implements:

- 1D CNN baseline for ECG classification  
- Preprocessing pipeline to convert raw ECG signals into numpy arrays  
- Difficulty scoring for curriculum learning  
- PDD-based training in three stages  
- Resume-based training across stages  

---

# Repository Structure

├── preprocess.py
├── train_1d_baseline.py
├── compute_difficulty.py
├── pdd_split.py
├── pdd_folder_generator.py
├── requirements.txt
├── .gitignore
└── README.md

results/baseline/
    X_train.npy
    y_train.npy
    X_val.npy
    y_val.npy
    X_test.npy
    y_test.npy

Baeline Train Model

python train_1d_baseline.py \
  --data_dir results/baseline \
  --epochs 30 \
  --batch_size 32 \
  --ckpt_dir baseline_ckpt \
  --augment


Create virtual environment

python -m venv venv
.\venv\Scripts\activate

Install required Python packages

pip install -r requirements.txt


Preprocess raw ECG signals

python preprocess.py ^
  --input_dir raw_data ^
  --meta_csv labels.csv ^
  --out_dir results/baseline ^
  --length 5000 ^
  --test_frac 0.1 ^
  --val_frac 0.1


Train the baseline 1D CNN-

python train_1d_baseline.py ^
  --data_dir results/baseline ^
  --epochs 30 ^
  --batch_size 32 ^
  --ckpt_dir baseline_ckpt ^
  --augment

Compute difficulty scores

python compute_difficulty.py

Create PDD (Stage1/2/3) splits-

python pdd_split.py

Generate final PDD training folders-

python pdd_folder_generator.py








