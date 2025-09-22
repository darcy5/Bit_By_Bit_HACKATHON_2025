import pandas as pd
import argparse

# ---------------------------
# Load CSV and extract all numbers
# ---------------------------
def load_csv_numbers(path):
    df = pd.read_csv(path, header=None, dtype=str)
    numbers = []
    for line in df[0]:
        # split by commas, remove extra whitespace
        parts = line.strip().split(",")
        for p in parts:
            try:
                numbers.append(float(p.strip()))
            except ValueError:
                pass  # ignore non-numeric
    return numbers

# ---------------------------
# Compare test vs model
# ---------------------------
def compare_numbers(test_numbers, model_numbers, tolerance=0.01):
    matches = 0
    for t in test_numbers:
        for m in model_numbers:
            if abs(t - m) / max(m, 1e-9) <= tolerance:  # avoid div by zero
                matches += 1
                break
    return matches

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare test CSV against model CSVs")
    parser.add_argument("--test", "-t", required=True, help="Path to test CSV file")
    args = parser.parse_args()

    # Load test CSV
    test_numbers = load_csv_numbers(args.test)

    # Model CSVs (hardcoded, unchanged)
    model_files = [
        "alexnet_data.csv",
        "densenet_data.csv",
        "resnet_data.csv",
        "vgg_data.csv",
        "alexnet_data_final.csv",
        "inception_v3_data.csv",
        "shufflenet_v2_x1_0_data.csv",
        "mobilenet_v2_data.csv",
    ]

    # Compare test against each model
    results = {}
    for f in model_files:
        model_numbers = load_csv_numbers(f)
        matches = compare_numbers(test_numbers, model_numbers, tolerance=0.01)
        percentage = matches / len(test_numbers) * 100
        results[f] = percentage
        print(f"{f}: {percentage:.2f}% numbers matched")

    # Best matching model
    best_model = max(results, key=results.get)
    print("\n=== Best Matching Model ===")
    print(f"Model: {best_model}")
    print(f"Match Percentage: {results[best_model]:.2f}%")

if __name__ == "__main__":
    main()