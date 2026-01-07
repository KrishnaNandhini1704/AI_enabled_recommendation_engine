import pandas as pd
import os

matrix_path = os.path.join("data", "processed", "user_item_matrix.csv")

if not os.path.exists(matrix_path):
    print(f"Error: Matrix file not found at {matrix_path}")
    exit(1)

print(f"Loading matrix from {matrix_path}...")
try:
    # Load with index_col=0 to assume first col is Customer ID
    df = pd.read_csv(matrix_path, index_col=0)
    print(f"Matrix Shape: {df.shape}")
    print(f"Index (Users): {df.index.nunique()} unique users")
    print(f"Columns (Items): {df.columns.nunique()} unique items")
    
    # Check sparsity
    total_cells = df.size
    non_zero_cells = (df != 0).sum().sum()
    sparsity = 1 - (non_zero_cells / total_cells)
    print(f"Sparsity: {sparsity:.4f}")
    
    # Sample values
    print("\nSample of non-zero interactions:")
    # Stack to get non-zero values easily
    stacked = df.stack()
    non_zeros = stacked[stacked != 0]
    print(non_zeros.head(5))
    
    print("\nMatrix Head:")
    print(df.head())

except Exception as e:
    print(f"Error inspecting matrix: {e}")
