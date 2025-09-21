import chipwhisperer as cw
import argparse

def connect_to_hardware():
    """Establishing connection to ChipWhisperer hardware"""
    try:
        scope = cw.scope()
        scope.default_setup()
        target = cw.target(scope)
        return scope, target
    except Exception as e:
        print(f"Error connecting to hardware: {e}")
        return None, None

def program_target(scope, hex_file):
    """Programming the target with the provided firmware"""
    try:
        prog = cw.programmers.STM32FProgrammer
        print("Programming target...")
        cw.program_target(scope, prog, hex_file)
        print("Programming done.")
        return True
    except Exception as e:
        print(f"Error programming target: {e}")
        return False

def verify_key(key_value, hex_file="simpleserial_rsa-CW308_STM32F3.hex"):
    """Verifying the recovered key by sending it as ciphertext"""
    # Connecting to hardware
    scope, target = connect_to_hardware()
    if scope is None or target is None:
        print("Failed to connect to hardware. Cannot verify key.")
        return False
    
    # Programming the target
    if not program_target(scope, hex_file):
        print("Failed to program target. Cannot verify key.")
        return False
    
    # Converting key to bytes
    key_bytes = key_value.to_bytes(2, 'big')
    
    # Sending key as ciphertext
    scope.arm()
    target.simpleserial_write('p', key_bytes)
    
    ret = scope.capture()
    if ret:
        print("Capture timed out during verification")
        scope.dis()
        target.dis()
        return False
    
    # Reading the response
    response = target.simpleserial_read('r', 2)
    plaintext = int.from_bytes(response, 'big')
    
    print(f"Verification: Sent key {key_value}, received plaintext {plaintext}")
    
    # Checking if we got the expected value
    success = plaintext == 6267
    
    # Disconnecting
    scope.dis()
    target.dis()
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Verify recovered RSA key")
    parser.add_argument("--verify-key", type=int, required=True, help="Key value to verify")
    parser.add_argument("--hex-file", default="E:\python\Warm up\Problem 1\simpleserial_rsa-CW308_STM32F3.hex", 
                       help="Firmware hex file")
    
    args = parser.parse_args()
    
    print(f"Verifying key: {args.verify_key}")
    print(f"Using firmware: {args.hex_file}")
    
    success = verify_key(args.verify_key, args.hex_file)
    
    if success:
        print("SUCCESS: Key is correct! The device returned 6267.")
    else:
        print("FAILED: Key verification failed. The device did not return 6267.")

if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                          # By Debopama and Sumantra
                                                                                                                                                         

#run in command prompt:  C:\Users\suman\AppData\Local\Programs\Python\Python310\python.exe "e:/python/Warm up/verification.py" --verify-key {key_value}