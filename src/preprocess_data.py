import pandas as pd
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    # The dataset might have multiple sheets, usually "Year 2009-2010" and "Year 2010-2011"
    # We will load both and concatenate them
    try:
        xls = pd.ExcelFile(filepath)
        print(f"Sheet names: {xls.sheet_names}")
        df1 = pd.read_excel(xls, 'Year 2009-2010')
        df2 = pd.read_excel(xls, 'Year 2010-2011')
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    print("Cleaning data...")
    # Drop rows with missing Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # Remove cancelled transactions (Invoice starts with 'C')
    # Ensure Invoice is string
    df['Invoice'] = df['Invoice'].astype(str)
    df = df[~df['Invoice'].str.startswith('C')]
    
    # Filter out invalid quantities or prices
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

def build_interaction_matrix(df):
    print("Building interaction matrix...")
    # formatting Customer ID
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # We can use Quantity as the interaction strength, or just binary purchase
    # Let's sum Quantity for each User-Item pair
    interactions = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()
    
    # Pivot
    matrix = interactions.pivot(index='Customer ID', columns='StockCode', values='Quantity')
    
    # Fill missing values with 0
    matrix = matrix.fillna(0)
    
    print(f"Matrix built. Shape: {matrix.shape}")
    return matrix

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "online_retail_II.xlsx")
    processed_dir = os.path.join("data", "processed")
    matrix_path = os.path.join(processed_dir, "user_item_matrix.csv")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
    else:
        df = load_data(raw_path)
        if df is not None:
            df_clean = clean_data(df)
            matrix = build_interaction_matrix(df_clean)
            
            print(f"Saving matrix to {matrix_path}...")
            matrix.to_csv(matrix_path)
            print("Done.")
