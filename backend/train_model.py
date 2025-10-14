import sys
import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else r"C:\quantum_trader\data\your_data.csv"
    data = load_data(csv)
    # ...existing code...
    # train_model(data)
    # ...existing code...