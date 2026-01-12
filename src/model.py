import numpy as np
from sklearn.decomposition import TruncatedSVD

class RecommendationModel:
    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_features = None
        self.item_features = None
        self.item_ids = None
        self.user_ids = None

    def train(self, matrix):
        """
        Trains the SVD model on the user-item matrix.
        matrix: pandas DataFrame with users as index and items as columns
        """
        self.item_ids = matrix.columns
        self.user_ids = matrix.index
        
        # Fit SVD
        print(f"Training SVD with {self.n_components} components...")
        self.user_features = self.svd.fit_transform(matrix)
        self.item_features = self.svd.components_
        
        print(f"Explained Variance Ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        return self

    def predict(self, user_id, item_id):
        """
        Predicts the interaction strength for a specific user and item.
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0  # Cold start problem: return 0 if unseen
        
        user_idx = self.user_ids.get_loc(user_id)
        item_idx = self.item_ids.get_loc(item_id)
        
        prediction = np.dot(self.user_features[user_idx], self.item_features[:, item_idx])
        return prediction

    def get_recommendations(self, user_id, n=5):
        """
        Get top N recommendations for a user.
        """
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.get_loc(user_id)
        
        # Reconstruct scores for this user: User vector dot Item matrix
        # user_vector (1, n_components) . item_matrix (n_components, n_items) -> (1, n_items)
        scores = np.dot(self.user_features[user_idx], self.item_features)
        
        # Get indices of top scores
        top_indices = scores.argsort()[::-1][:n]
        
        recommendations = [self.item_ids[i] for i in top_indices]
        return recommendations
