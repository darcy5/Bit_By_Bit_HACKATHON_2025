import numpy as np
import csv

# --- CONFIGURATION ---
NUM_BITS = 15
KNOWN_MSB = 1
KNOWN_LSBS = [0, 0, 0, 1]  # Last 4 bits
UNKNOWN_BITS_START = 1
UNKNOWN_BITS_END = 11       # Only attack these 10 unknown bits
RSA_N = 64507
TRACE_FILE = "traces.csv"
TRACE_LENGTH = 5000          # Samples per trace (adjust if different)
SEARCH_AFTER_SQUARE = 20      # Window after square to correlate multiply
WINDOW_LEN = 15               # Number of samples in correlation window

# --- HELPER FUNCTIONS ---
def load_traces(csv_file):
    """Loading traces and ciphertexts from CSV"""
    traces = []
    cts = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            cts.append(int(row[0]))
            traces.append([float(x) for x in row[1:]])
    return np.array(traces), np.array(cts)

def hamming_weight(val):
    """Hamming weight of integer"""
    return bin(val).count('1')

def simulate_intermediate(ct, key_bits, bit_pos, hypothesis):
    """
    Simulating intermediate modular exponentiation value up to current bit
    Using Python int to avoid overflow
    """
    val = 1
    ct = int(ct)
    for i in range(bit_pos + 1):
        bit = key_bits[i] if i < bit_pos else hypothesis
        val = (val * val) % RSA_N   # Square always
        if bit == 1:
            val = (val * ct) % RSA_N
    return val

def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

# --- LOAD TRACES ---
traces, ciphertexts = load_traces(TRACE_FILE)
num_traces, trace_len = traces.shape
print(f"Loaded {num_traces} traces of length {trace_len}")

# --- INITIALIZE KEY ---
key_guess = [0]*NUM_BITS
key_guess[0] = KNOWN_MSB
key_guess[-4:] = KNOWN_LSBS

# --- CPA ATTACK ---
for bit_pos in range(UNKNOWN_BITS_START, UNKNOWN_BITS_END):
    print(f"Recovering bit {bit_pos} ...")
    
    hw0 = np.array([hamming_weight(simulate_intermediate(ct, key_guess, bit_pos, 0)) for ct in ciphertexts])
    hw1 = np.array([hamming_weight(simulate_intermediate(ct, key_guess, bit_pos, 1)) for ct in ciphertexts])
    
    # Point selection: choose a window where multiply occurs
    start_idx = bit_pos * 30 + SEARCH_AFTER_SQUARE
    end_idx = min(start_idx + WINDOW_LEN, trace_len)
    
    corr0 = np.array([np.corrcoef(hw0, traces[:, i])[0,1] for i in range(start_idx, end_idx)])
    corr1 = np.array([np.corrcoef(hw1, traces[:, i])[0,1] for i in range(start_idx, end_idx)])
    
    max0 = np.max(np.abs(corr0))
    max1 = np.max(np.abs(corr1))
    
    guessed_bit = 1 if max1 > max0 else 0
    key_guess[bit_pos] = guessed_bit
    
    print(f"Bit {bit_pos} recovered as {guessed_bit} (corr0={max0:.4f}, corr1={max1:.4f})")

# --- FINAL KEY ---
recovered_key = bits_to_int(key_guess)
print(f"\nRecovered 15-bit key: {recovered_key} (binary: {''.join(map(str,key_guess))})")                                                                                                                  # By Debopama and Sumantra                                                                                                                                                               # By Debopama and Sumantra


