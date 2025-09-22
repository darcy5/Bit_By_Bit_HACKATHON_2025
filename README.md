# Bit_By_Bit_HACKATHON_2025

**Team Destra**

This repository contains the code, output files, reports of three problems given in hackathon.

## Problem 1
**Repository Name:** https://github.com/darcy5/Bit_By_Bit_HACKATHON_2025/tree/main/Problem1_HACKATHON

This folder contains:
1. Problem Statement file: `Hackathon_2025_problem1.pdf`  
2. Hex Code File: `simpleserial_rsa-CW308_STM32F3.hex` — # We observed the power trace on running this hex file  
3. Trace Catcher: `trace_collector.py` — # This python script collected the trace  
4. Trace Storage: `traces.csv`  
5. `rsa_response_plaintext.csv` — # This file stores the returned plaintext for each trace against the ciphertext  
6. `CPA_attack.py` — # The ultimate side channel attack script, which finds the key  
7. `verification.py` — # This script finds the plaintext against the key and verifies if the plaintext is correct or not  
8. Report: `Hackathon Problem 1 Report.pdf`  
9. `Result of problem 1.txt`


## Problem 2
**Repository Name:** https://github.com/darcy5/Bit_By_Bit_HACKATHON_2025/tree/main/Problem2_HACKATHON

This folder contains:
1. `IP Details.txt`  
2. `Perf Commands.txt` — # contains perf commands used to collect traces based on different events  
3. `Problem2_HACKATHON_Report.pdf`  
4. `prediction.py` — # comparing the obtained traces with the provided traces of different models  
5. `trace_cleaner.py`  
6. `Result of problem 2.txt`


## Problem 3
**Repository Name:** https://github.com/darcy5/Bit_By_Bit_HACKATHON_2025/tree/main/Problem3_HACKATHON

This folder contains:
1. `DLSCA_Hackathon_Question.pdf`  
2. `Hackathon_Problem_3_Report.pdf`  
3. `all_sorted.py`  
4. `all_sorted_keys.txt`  
5. `datasetA.npz`  
6. `datasetB.npz`  
7. `model_saved.h5`  
8. `scores.npy`  
9. `sol.py` — # Ultimate Solution scripts; Run Using:  
   ```bash
   python sol.py --profiling datasetB.npz --target datasetA.npz --label_type sbox --epochs 80




-By Sumantra Dutta and Debopama Basu
