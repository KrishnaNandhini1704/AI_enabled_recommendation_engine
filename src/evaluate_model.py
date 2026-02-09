import pandas as pd
import numpy as np
import os
from model import RecommendationModel
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

def load_matrix(filepath):
    print(f"Loading matrix from {filepath}...")
    df = pd.read_csv(filepath, index_col=0)
    df = df.fillna(0)
    # Apply log1p normalization to handle skewed data
    df = np.log1p(df)
    return df

def train_test_split(matrix, test_ratio=0.2, random_state=42):
    """
    Splits the interaction matrix into train and test sets by masking some interactions.
    We act as if the masked interactions in the test set never happened during training.
    Returns:
        train_matrix: DataFrame with masked values set to 0
        test_set: List of (user, item, actual_value) tuples
    """
    np.random.seed(random_state)
    
    # Create a copy for training (masked) and testing (ground truth of masked items)
    train_matrix = matrix.copy()
    test_set = [] # List of (user, item, value) tuples that are masked
    
    # Stack to get (user, item, value)
    unstacked = matrix.stack()
    interactions = unstacked[unstacked > 0]
    
    # Sample indices
    test_indices = np.random.choice(interactions.index, size=int(len(interactions) * test_ratio), replace=False)
    
    for (user, item) in test_indices:
        actual_value = train_matrix.loc[user, item]
        train_matrix.loc[user, item] = 0 # Mask in training
        test_set.append((user, item, actual_value))
        
    print(f"Split completed. Masked {len(test_set)} interactions for testing.")
    return train_matrix, test_set

def get_metrics_at_k(model, train_matrix, test_set, k=5):
    """
    Evaluates Precision@K, Recall@K, F1@K, and Hit Rate@K (Accuracy).
    test_set is a list of (user, item, actual_value) tuples that were hidden.
    """
    
    # Group test items by user
    test_user_items = {}
    for user, item, _ in test_set:
        if user not in test_user_items:
            test_user_items[user] = []
        test_user_items[user].append(item)
        
    precisions = []
    recalls = []
    hits = []
    
    for user, ground_truth_items in test_user_items.items():
        if user not in model.user_ids:
            continue
            
        user_idx = model.user_ids.get_loc(user)
        scores = np.dot(model.user_features[user_idx], model.item_features)
        
        # Filter training items
        known_items = train_matrix.loc[user]
        known_indices = [model.item_ids.get_loc(item) for item in known_items[known_items > 0].index if item in model.item_ids]
        
        scores[known_indices] = -np.inf
        
        # Get top K
        top_indices = scores.argsort()[::-1][:k]
        top_items = [model.item_ids[i] for i in top_indices]
        
        # Calculate Metrics
        n_rel = len(set(top_items).intersection(ground_truth_items))
        
        precision = n_rel / k
        recall = n_rel / len(ground_truth_items) if len(ground_truth_items) > 0 else 0
        hit = 1 if n_rel > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)
        
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    accuracy_hit_rate = np.mean(hits) if hits else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f1, accuracy_hit_rate

def calculate_rmse(model, test_set):
    """
    Calculates RMSE between predicted scores and actual values for the masked test set.
    """
    y_true = []
    y_pred = []
    
    for user, item, actual_value in test_set:
        pred_value = model.predict(user, item)
        y_true.append(actual_value)
        y_pred.append(pred_value)
        
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def evaluate_metrics(model, train_matrix, test_set):
    print("Evaluating metrics...")
    
    # Calculate RMSE
    rmse = calculate_rmse(model, test_set)
    print(f"Test RMSE:   {rmse:.4f}")
    
    # Calculate Top-K metrics
    p5, r5, f1_5, acc5 = get_metrics_at_k(model, train_matrix, test_set, k=5)
    print(f"Precision@5: {p5:.4f}")
    print(f"Recall@5:    {r5:.4f}")
    print(f"F1@5:        {f1_5:.4f}")
    print(f"Accuracy@5:  {acc5:.4f} (Hit Rate)")
    
    p10, r10, f1_10, acc10 = get_metrics_at_k(model, train_matrix, test_set, k=10)
    print(f"Precision@10: {p10:.4f}")
    print(f"Recall@10:    {r10:.4f}")
    print(f"F1@10:        {f1_10:.4f}")
    print(f"Accuracy@10:  {acc10:.4f} (Hit Rate)")
    
    return f1_10

def tune_model(matrix):
    print("\n--- Starting Hyperparameter Tuning ---")
    
    # Split data once
    train_matrix, test_set = train_test_split(matrix, test_ratio=0.2)
    
    best_score = -1
    best_n = 5
    
    # Try different components
    for n in [50, 100, 150]:
        print(f"\nTraining with n_components={n}...")
        model = RecommendationModel(n_components=n, random_state=42)
        model.train(train_matrix)
        
        # Metric: Let's optimize for F1@10
        print(f"Evaluating n_components={n}...")
        f1 = evaluate_metrics(model, train_matrix, test_set)
        
        if f1 > best_score:
            best_score = f1
            best_n = n
            
    print(f"\nBest Configuration: n_components={best_n} with F1@10={best_score:.4f}")
    return best_n

if __name__ == "__main__":
    matrix_path = os.path.join("data", "processed", "user_item_matrix.csv")
    if not os.path.exists(matrix_path):
        print("Matrix not found. Run preprocessing first.")
    else:
        matrix = load_matrix(matrix_path)
        tune_model(matrix)
