import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clean branch misses CSV file")
    parser.add_argument("--input", "-i", required=True, help="Input raw perf CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output cleaned CSV file")

    args = parser.parse_args()

    # Reading CSV, skip comment lines starting with '#'
    df = pd.read_csv(args.input, comment='#', delim_whitespace=True, header=None)

    # Extracting the counts column (second column, index 1)
    counts = df[1]

    # Saving only the counts to a new CSV
    counts.to_csv(args.output, index=False, header=False)

    print(f"Cleaned instructions saved to {args.output}")
    print(counts.head(20))

if __name__ == "__main__":
    main()