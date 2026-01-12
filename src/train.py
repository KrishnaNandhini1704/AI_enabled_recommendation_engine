import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error
from model import RecommendationModel

def load_matrix(filepath):
    print(f"Loading matrix from {filepath}...")
    # Read csv, set Customer ID as index
    df = pd.read_csv(filepath, index_col=0)
    # Fill NaN with 0 just in case
    df = df.fillna(0)
    return df

def evaluate_rmse(model, matrix):
    """
    Evaluates the model using Reconstruction RMSE on the training set.
    """
    reconstructed_matrix = np.dot(model.user_features, model.item_features)
    
    mse = mean_squared_error(matrix.values, reconstructed_matrix)
    rmse = np.sqrt(mse)
    return rmse

def main():
    data_dir = os.path.join("data", "processed")
    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    matrix_path = os.path.join(data_dir, "user_item_matrix.csv")
    
    if not os.path.exists(matrix_path):
        print(f"Error: {matrix_path} not found. Run preprocessing first.")
        return

    # Load Data
    matrix = load_matrix(matrix_path)
    print(f"Matrix shape: {matrix.shape}")
    
    # Initialize Model
    # Using 50 components as a starting point.
    model = RecommendationModel(n_components=50) 
    
    # Train
    model.train(matrix)
    
    # Evaluate (Reconstruction Error)
    rmse = evaluate_rmse(model, matrix)
    print(f"Training Reconstruction RMSE: {rmse:.4f}")
    
    # Save Model
    model_path = os.path.join(models_dir, "svd_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Test a prediction
    # Just grab the first user and item to verify it works
    if not matrix.empty:
        sample_user = matrix.index[0]
        sample_item = matrix.columns[0]
        try:
            pred = model.predict(sample_user, sample_item)
            actual = matrix.loc[sample_user, sample_item]
            print(f"Sample Check - User: {sample_user}, Item: {sample_item}")
            print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")
            
            # Get recommendations
            recs = model.get_recommendations(sample_user, n=3)
            print(f"Top 3 Recommendations for user {sample_user}: {recs}")
            
        except Exception as e:
            print(f"Error during sample check: {e}")

if __name__ == "__main__":
    main()
