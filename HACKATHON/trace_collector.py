import chipwhisperer as cw
import numpy as np
import random
import csv

HEXFILE = "simpleserial_rsa-CW308_STM32F3.hex" 
RSA_N = 64507
NUM_TRACES = 5000        
SAMPLES = 5000        

def program_target(scope):
    prog = cw.programmers.STM32FProgrammer
    print("[*] Programming target ...")
    cw.program_target(scope, prog, HEXFILE)
    print("[*] Programming done.")

def init_scope():
    scope = cw.scope()
    scope.default_setup()
    scope.clock.adc_src = "clkgen_x1"
    scope.adc.samples = SAMPLES
    return scope

def capture_traces(scope, target, num_traces=NUM_TRACES, out_csv="traces.csv"):
    random.seed(0xCAFEBABE)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(num_traces):
            # pick ciphertext < N
            c_int = random.randint(0, RSA_N - 1)
            ct_bytes = c_int.to_bytes(2, "big")

            scope.arm()
            target.simpleserial_write('p', ct_bytes)

            ret = scope.capture()
            if ret:
                print(f"[!] Timeout on trace {i+1}, skipping")
                continue

            trace = scope.get_last_trace()  
            trace = np.array(trace)     

            # Row details: ciphertext first, then all samples
            writer.writerow([c_int] + trace.tolist())

            if (i + 1) % 50 == 0:
                print(f"[*] Captured {i+1} traces")

    print(f"[*] Saved {num_traces} traces to {out_csv}")

if __name__ == "__main__":
    scope = init_scope()
    target = cw.target(scope)

    program_target(scope)

    # Starting capture
    capture_traces(scope, target, NUM_TRACES, out_csv="traces.csv")