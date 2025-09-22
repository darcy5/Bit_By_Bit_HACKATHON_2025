#!/usr/bin/env python3
"""
sca_profile_and_attack.py

End-to-end profiling + cross-device key recovery for AES first-byte using a 1D-CNN.

Requirements:
  pip install numpy tensorflow scikit-learn

Usage:
  python sca_profile_and_attack.py \
    --profiling datasetB.npz \
    --target datasetA.npz \
    --label_type sbox \
    --crop_start 0 --crop_end 0 \
    --epochs 80

Outputs:
  model_saved.h5
  sorted_keys.npy
  scores.npy
  sorted_keys.txt
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape


# --------------------------
# AES Sbox and helpers
# --------------------------
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

def hw8(x):
    """Hamming weight 0..8 for a byte or numpy array of bytes"""
    return np.unpackbits(np.array(x, dtype=np.uint8).reshape(-1,1), axis=1).sum(axis=1)

# --------------------------
# Data loading / preprocessing
# --------------------------

def load_npz(path):
    """
    Expects .npz with 'trace' (n x L), 'plaintext' (n,) and optionally 'key' (n,) for profiling.
    If it's a single .npz containing separate .npy files, np.load handles it too.
    """
    data = np.load(path, allow_pickle=True)
    # try common keys
    trace = None
    plaintext = None
    key = None
    for k in data.files:
        if 'trace' in k.lower():
            trace = data[k]
        elif 'plain' in k.lower():
            plaintext = data[k]
        elif 'key' in k.lower():
            key = data[k]
    # fallback if file itself is a dict of separate npy's
    if trace is None and 'trace.npy' in os.listdir(os.path.dirname(path) or '.'):
        trace = np.load(os.path.join(os.path.dirname(path), 'trace.npy'))
    return trace, plaintext, key

def preprocess_traces(traces, crop_start=0, crop_end=0, baseline_subtract=True, mean=None, std=None):
    """
    traces: numpy array shape (n, L)
    crop_start,crop_end: ints. If crop_end==0 -> keep to end.
    baseline_subtract: subtract mean of each trace
    mean,std: if provided, standardize using these; otherwise compute and return them.
    Returns: preproc_traces, mean, std
    """
    if traces.ndim != 2:
        raise ValueError("Traces should be 2D (n_traces, length). Got shape: " + str(traces.shape))
    n, L = traces.shape
    start = crop_start
    end = L - crop_end if crop_end > 0 else L
    traces = traces[:, start:end].astype(np.float32)

    if baseline_subtract:
        # subtract per-trace mean
        traces = traces - traces.mean(axis=1, keepdims=True)

    if mean is None or std is None:
        mean = traces.mean()
        std = traces.std()
        # avoid zero
        if std == 0:
            std = 1.0
    traces = (traces - mean) / (std + 1e-12)
    return traces, mean, std

# --------------------------
# Labels for profiling
# --------------------------
def compute_labels_sbox(plaintext_arr, key_arr):
    # both arrays shape (n,)
    xx = np.bitwise_xor(plaintext_arr.astype(np.uint8), key_arr.astype(np.uint8))
    # use SBOX table
    return SBOX[xx]

def compute_labels_hw_sbox(plaintext_arr, key_arr):
    s = compute_labels_sbox(plaintext_arr, key_arr)
    return hw8(s)

# --------------------------
# Model
# --------------------------
def build_cnn(input_length, num_classes):
    inp = Input(shape=(input_length,))
    x = Reshape((input_length, 1))(inp)   # replaces tf.expand_dims

    x = Conv1D(32, 11, activation='relu', padding='same')(x)
    x = Conv1D(64, 11, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)

# --------------------------
# Scoring (aggregation)
# --------------------------
def compute_scores_proba(probas_A, plaintexts_A, label_type='sbox', eps=1e-40):
    """
    probas_A: numpy array (Na, num_classes) predicted from model for each A trace
    plaintexts_A: (Na,) plaintext byte values
    label_type: 'sbox' or 'hw'
    returns scores array shape (256,)
    """
    Na, numc = probas_A.shape
    scores = np.zeros(256, dtype=np.float64)

    # For each candidate key
    for k in range(256):
        # compute z_i,k for each trace
        if label_type == 'sbox':
            z = SBOX[np.bitwise_xor(plaintexts_A.astype(np.uint8), np.uint8(k))]
            # z in 0..255, must be index into probas_A columns (numc should be 256)
        else:
            s = SBOX[np.bitwise_xor(plaintexts_A.astype(np.uint8), np.uint8(k))]
            z = hw8(s)  # 0..8

        # ensure indices are integers and in range
        z = z.astype(np.int64)
        # pick the probabilities p_{i}[z_i]
        pz = probas_A[np.arange(Na), z]  # shape (Na,)
        # clamp
        pz = np.maximum(pz, eps)
        # sum log-likelihood
        scores[k] = np.sum(np.log(pz.astype(np.float64)))
    return scores

# --------------------------
# Main training + attack flow
# --------------------------
def main(args):
    # 1. Load datasets
    print("Loading profiling:", args.profiling)
    traces_B, plaintext_B, key_B = load_npz(args.profiling)
    if traces_B is None or plaintext_B is None or key_B is None:
        raise RuntimeError("Profiling dataset must contain trace, plaintext, and key")
    print("Profiling shapes:", traces_B.shape, plaintext_B.shape, key_B.shape)
    print("Loading target:", args.target)
    traces_A, plaintext_A, _ = load_npz(args.target)
    if traces_A is None or plaintext_A is None:
        raise RuntimeError("Target dataset must contain trace and plaintext")
    print("Target shapes:", traces_A.shape, plaintext_A.shape)

    # 2. Preprocess (crop + baseline + standardize using profiling stats)
    print("Preprocessing (crop_start=%d crop_end=%d)..." % (args.crop_start, args.crop_end))
    traces_B_proc, meanB, stdB = preprocess_traces(traces_B, crop_start=args.crop_start, crop_end=args.crop_end, baseline_subtract=True, mean=None, std=None)
    traces_A_proc, _, _ = preprocess_traces(traces_A, crop_start=args.crop_start, crop_end=args.crop_end, baseline_subtract=True, mean=meanB, std=stdB)
    print("Processed lengths: B:", traces_B_proc.shape, "A:", traces_A_proc.shape)

    # 3. Labels
    if args.label_type == 'sbox':
        labels_B = compute_labels_sbox(plaintext_B, key_B)   # 0..255
        num_classes = 256
    elif args.label_type == 'hw':
        labels_B = compute_labels_hw_sbox(plaintext_B, key_B)  # 0..8
        num_classes = 9
    else:
        raise ValueError("Unknown label_type")

    # 4. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(traces_B_proc, labels_B, test_size=args.val_split, random_state=42, stratify=labels_B)
    print("Train/Val sizes:", X_train.shape[0], X_val.shape[0])

    # 5. Build model
    input_length = X_train.shape[1]
    model = build_cnn(input_length, num_classes)
    model.compile(optimizer=Adam(args.lr), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()

    # callbacks
    ckpt_path = args.model_out or 'model_saved.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]

    # 6. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
    )

    # 7. Save final model
    model.save(ckpt_path)
    print("Saved model to", ckpt_path)

    # 8. Predict on A (probas)
    print("Predicting probabilities on target traces...")
    probas_A = model.predict(traces_A_proc, batch_size=args.batch_size, verbose=1)  # shape (Na, num_classes)
    print("Probas shape:", probas_A.shape)

    # 9. Compute scores for all 256 keys
    print("Computing scores...")
    scores = compute_scores_proba(probas_A, plaintext_A, label_type=args.label_type, eps=1e-40)
    # sort descending
    sorted_indices = np.argsort(-scores)
    sorted_keys = sorted_indices  # keys most likely first
    print("Top 10 keys (most→less likely):", sorted_keys[:10].tolist())

    # Save results
    np.save('scores.npy', scores)
    np.save('sorted_keys.npy', sorted_keys)
    with open('sorted_keys.txt', 'w') as f:
        f.write("sorted_keys_most_to_least_likely:\n")
        f.write(", ".join([str(int(k)) for k in sorted_keys.tolist()]) + "\n")
        f.write("top10: " + ", ".join([str(int(k)) for k in sorted_keys[:10].tolist()]) + "\n")

    # If user knows true key for A (optional) print rank
    if hasattr(args, 'true_key') and args.true_key is not None:
        true_key = int(args.true_key)
        rank = int(np.where(sorted_keys == true_key)[0][0])
        print("True key", true_key, "rank:", rank)
    else:
        print("If you know the true key, pass --true_key to see its rank.")

    print("Done. Saved outputs: model, scores.npy, sorted_keys.npy, sorted_keys.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiling', type=str, required=True, help='Path to profiling npz (datasetB.npz)')
    parser.add_argument('--target', type=str, required=True, help='Path to target npz (datasetA.npz)')
    parser.add_argument('--label_type', type=str, default='sbox', choices=['sbox','hw'], help='sbox (256 classes) or hw (9 classes)')
    parser.add_argument('--crop_start', type=int, default=0, help='Crop start samples (0-based)')
    parser.add_argument('--crop_end', type=int, default=0, help='Crop end samples (count from right); 0 means keep to end')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--model_out', type=str, default='model_saved.h5')
    parser.add_argument('--true_key', type=int, default=None)
    args = parser.parse_args()
    main(args)
