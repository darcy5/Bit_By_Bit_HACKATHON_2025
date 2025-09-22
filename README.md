# Bit_By_Bit_HACKATHON_2025

This repository contains the code, output files, reports of three problems given in hackathon.

Problem 1:
Repository Name: https://github.com/darcy5/Bit_By_Bit_HACKATHON_2025/tree/main/Problem1_HACKATHON

This folder contains:
  Problem Statement file: Hackathon_2025_problem1.pdf
  Hex Code File: simpleserial_rsa-CW308_STM32F3.hex     # We observed the power trace on running this hex file
  Trace Catcher: trace_collector.py                     # This python script collected the trace
  Trace Storage: traces.csv
  rsa_response_plaintext.csv                            # This file stores the returned plaintext for each trace against the ciphertext
  CPA_attack.py                                         # The ultimate side channel attack script, which finds the key
  verification.py                                       # This script finds the plaintext against the key and varifies if the plaintext is correct or not
  Report: Hackathon_Problem_1_Report.pdf



  Problem 2:
  Repository Name:

  Problem 3:s

  This folder contains:
    1.  DLSCA_Hackathon_Question.pdf
    2.  Hackathon_Problem_3_Report.pdf
    3.  all_sorted.py
    4.  all_sorted_keys.txt
    5.  datasetA.npz
    6.  datasetB.npz
    7.  model_saved.h5
    8.  scores.npy
    9.  sol.py                 # Ultimate Solution scripts; Run Using: python sol.py --profiling datasetB.npz --target datasetA.npz --label_type sbox --epochs 80
    10. sorted_keys.npy
    11. sorted_keys.txt
