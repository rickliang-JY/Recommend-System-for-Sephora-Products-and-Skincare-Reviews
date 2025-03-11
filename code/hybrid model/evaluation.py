#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix
#%% Recommender Evaluation
class RecommenderEvaluation:
    """
    A class for evaluating recommender system performance focusing on diversity and personalization.
    """
    
    def __init__(self, products_df, reviews_df, user_product_matrix, hybrid_sim, char_matrix_data, 
                 user_characteristics_df, recommend_products_func, recommend_for_user_func):
        """
        Initialize the evaluation class with necessary data and functions.
        """
        self.products_df = products_df
        self.reviews_df = reviews_df
        self.user_product_matrix = user_product_matrix
        self.hybrid_sim = hybrid_sim
        self.char_matrix_data = char_matrix_data
        self.user_characteristics_df = user_characteristics_df
        self.recommend_products = recommend_products_func
        self.recommend_for_user = recommend_for_user_func
        
        # Metrics dictionary to store results
        self.metrics = {}
    
    def evaluate_diversity(self, n_recommendations=10):
        """
        Evaluate the diversity of recommendations.
        This measures how different the recommendations are from each other.
        
        Parameters:
        -----------
        n_recommendations : int, default=10
            Number of recommendations to generate for each evaluation
        """
        print("Evaluating recommendation diversity...")
        
        # Sample different products and characteristic combinations
        n_samples = 50
        diversity_scores = []
        
        # Sample products
        sampled_products = np.random.choice(self.products_df['product_id'], 
                                           min(n_samples, len(self.products_df)), 
                                           replace=False)
        
        for product_id in sampled_products:
            # Get recommendations
            try:
                recommendations = self.recommend_products(product_id=product_id, 
                                                          num_recommendations=n_recommendations)
                
                if len(recommendations) < 2:
                    continue
                
                # Calculate average pairwise similarity between recommended products
                rec_ids = recommendations['product_id'].tolist()
                similarities = []
                
                for i in range(len(rec_ids)):
                    for j in range(i+1, len(rec_ids)):
                        if rec_ids[i] in self.hybrid_sim.index and rec_ids[j] in self.hybrid_sim.columns:
                            sim = self.hybrid_sim.loc[rec_ids[i], rec_ids[j]]
                            similarities.append(sim)
                
                if similarities:
                    # Calculate diversity as 1 - average similarity
                    avg_similarity = np.mean(similarities)
                    diversity = 1 - avg_similarity
                    diversity_scores.append(diversity)
                    
            except Exception as e:
                print(f"Error evaluating diversity for product {product_id}: {e}")
                continue
        
        # Calculate overall diversity
        mean_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        self.metrics['recommendation_diversity'] = mean_diversity
        print(f"Average recommendation diversity: {mean_diversity:.4f}")
        return mean_diversity
    
    def evaluate_personalization(self, n_users=20, n_recommendations=10):
        """
        Evaluate the personalization of recommendations.
        This measures how different the recommendations are across users.
        
        Parameters:
        -----------
        n_users : int, default=20
            Number of users to sample for evaluation
        n_recommendations : int, default=10
            Number of recommendations to generate for each user
        """
        print("Evaluating personalization...")
        
        # Sample users
        all_users = list(self.user_product_matrix.columns)
        if len(all_users) <= n_users:
            sampled_users = all_users
        else:
            sampled_users = np.random.choice(all_users, n_users, replace=False)
        
        # Get recommendations for each user
        user_recommendations = {}
        for user_id in sampled_users:
            try:
                recommendations = self.recommend_for_user(user_id, num_recommendations=n_recommendations)
                user_recommendations[user_id] = set(recommendations['product_id'])
            except:
                # Skip users that may not have enough ratings
                continue
        
        # Calculate pairwise Jaccard dissimilarity
        if len(user_recommendations) < 2:
            print("Not enough valid users for personalization evaluation")
            return 0
        
        jaccard_dissimilarities = []
        
        users = list(user_recommendations.keys())
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                set_i = user_recommendations[users[i]]
                set_j = user_recommendations[users[j]]
                
                # Jaccard similarity = |intersection| / |union|
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                
                if union > 0:
                    # Jaccard dissimilarity = 1 - Jaccard similarity
                    dissimilarity = 1 - (intersection / union)
                    jaccard_dissimilarities.append(dissimilarity)
        
        # Calculate average dissimilarity (higher means more personalized)
        mean_dissimilarity = np.mean(jaccard_dissimilarities) if jaccard_dissimilarities else 0
        
        self.metrics['personalization'] = mean_dissimilarity
        print(f"Personalization score: {mean_dissimilarity:.4f}")
        return mean_dissimilarity
    
    def evaluate_novelty(self, n_users=20, n_recommendations=10):
        """
        Evaluate the novelty of recommendations.
        This measures how popular (inverse of novelty) the recommended items are on average.
        
        Parameters:
        -----------
        n_users : int, default=20
            Number of users to sample for evaluation
        n_recommendations : int, default=10
            Number of recommendations to generate for each user
        """
        print("Evaluating novelty...")
        
        # Calculate popularity of each product (normalized)
        product_counts = self.reviews_df['product_id'].value_counts()
        total_reviews = len(self.reviews_df)
        product_popularity = product_counts / total_reviews
        
        # Sample users
        all_users = list(self.user_product_matrix.columns)
        if len(all_users) <= n_users:
            sampled_users = all_users
        else:
            sampled_users = np.random.choice(all_users, n_users, replace=False)
        
        # Get recommendations for each user and calculate the average popularity
        novelty_scores = []
        
        for user_id in sampled_users:
            try:
                recommendations = self.recommend_for_user(user_id, num_recommendations=n_recommendations)
                rec_products = recommendations['product_id'].tolist()
                
                # Get popularity of recommended products
                rec_popularity = [product_popularity.get(pid, 0) for pid in rec_products]
                
                # Novelty is inverse of popularity (log scale to reduce impact of very popular items)
                if rec_popularity:
                    avg_popularity = np.mean(rec_popularity)
                    novelty = -np.log(avg_popularity + 1e-10)  # Add small epsilon to avoid log(0)
                    novelty_scores.append(novelty)
            except:
                # Skip users that may not have enough ratings
                continue
        
        # Calculate average novelty
        mean_novelty = np.mean(novelty_scores) if novelty_scores else 0
        
        self.metrics['novelty'] = mean_novelty
        print(f"Novelty score: {mean_novelty:.4f}")
        return mean_novelty
    
    def evaluate_key_metrics(self):
        """
        Run only the key evaluation metrics and return the results.
        """
        print("Evaluating key recommender system metrics...")
        
        # Run only the diversity, personalization, and novelty metrics
        self.evaluate_diversity()
        self.evaluate_personalization()
        self.evaluate_novelty()
        
        print("\nEvaluation complete!")
        return self.metrics
    
    def plot_key_metrics(self):
        """
        Generate visualizations of the key evaluation metrics.
        """
        if not self.metrics:
            print("No metrics to plot. Run evaluate_key_metrics first.")
            return
        
        # Set up the matplotlib figure
        plt.figure(figsize=(10, 6))
        
        # Plot diversity, personalization, and novelty
        if all(k in self.metrics for k in ['recommendation_diversity', 'personalization', 'novelty']):
            # Normalize novelty to 0-1 scale for visualization
            novelty = self.metrics['novelty']
            max_novelty = 10  # Assuming log scale, this is a reasonable max
            normalized_novelty = min(novelty / max_novelty, 1)
            
            metrics = ['Diversity', 'Personalization', 'Novelty (Norm.)']
            values = [
                self.metrics['recommendation_diversity'], 
                self.metrics['personalization'],
                normalized_novelty
            ]
            
            # Create bar chart
            bars = plt.bar(metrics, values, color=['green', 'orange', 'purple'])
            plt.ylabel('Score (0-1)')
            plt.title('Key Recommender System Metrics')
            plt.ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
            
            # Add actual novelty value as text
            plt.figtext(0.7, 0.02, f"Raw Novelty: {self.metrics['novelty']:.3f}", 
                       ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.savefig('recommender_key_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt

def evaluate_recommender_key_metrics(products_csv, review_data_csv, hybrid_similarity_pkl, 
                                   char_matrix_data_pkl, user_characteristics_csv, user_product_matrix_pkl=None):
    """
    Load data and evaluate the key metrics of the recommender system.
    
    Parameters:
    -----------
    products_csv : str
        Path to products CSV file
    review_data_csv : str
        Path to review data CSV file
    hybrid_similarity_pkl : str
        Path to hybrid similarity matrix pickle file
    char_matrix_data_pkl : str
        Path to characteristic matrix data pickle file
    user_characteristics_csv : str
        Path to user characteristics CSV file
    user_product_matrix_pkl : str, optional
        Path to user-product matrix pickle file
        
    Returns:
    --------
    RecommenderEvaluation object with evaluation results
    """
    print("Loading data for evaluation...")
    
    # Load CSV files
    products_df = pd.read_csv(products_csv)
    reviews_df = pd.read_csv(review_data_csv)
    user_characteristics_df = pd.read_csv(user_characteristics_csv)
    
    # Load pickle files
    with open(hybrid_similarity_pkl, 'rb') as f:
        hybrid_sim = pickle.load(f)
    
    with open(char_matrix_data_pkl, 'rb') as f:
        char_matrix_data = pickle.load(f)
    
    # Define function to create user_product_matrix
    def create_user_product_matrix(reviews_df):
        """
        Create a user-product matrix from review data.
        """
        print("Creating user-product matrix from reviews...")
        # Aggregate by taking the mean rating per (product_id, author_id)
        reviews_agg = reviews_df.groupby(["product_id", "author_id"], as_index=False).agg({"rating": "mean"})
        
        # Now pivot the table
        user_product_matrix = reviews_agg.pivot(index="product_id", columns="author_id", values="rating")
        
        # Fill missing values with 0 (unrated products)
        user_product_matrix = user_product_matrix.fillna(0)
        
        return user_product_matrix
    
    # Load user-product matrix if provided, otherwise create from reviews
    user_product_matrix = None
    if user_product_matrix_pkl:
        try:
            with open(user_product_matrix_pkl, 'rb') as f:
                user_product_matrix = pickle.load(f)
            print(f"Successfully loaded user-product matrix from {user_product_matrix_pkl}")
        except Exception as e:
            print(f"Could not load user-product matrix from {user_product_matrix_pkl}. Creating from reviews.")
            print(f"Error: {e}")
            user_product_matrix = create_user_product_matrix(reviews_df)
    
    if user_product_matrix is None:
        user_product_matrix = create_user_product_matrix(reviews_df)
    
    # Initialize cache for characteristic similarity
    characteristic_cache = {}
    
    # Define recommendation functions
    def recommend_products_func(product_id=None, user_characteristics=None, num_recommendations=5, alpha=0.7, beta=0.5):
        """
        Recommends products based on a hybrid approach combining product similarity and user characteristics.
        """
        if product_id is None and user_characteristics is None:
            raise ValueError("Either product_id or user_characteristics must be provided")
        
        # If product_id is provided, get hybrid similarity scores
        if product_id is not None:
            # Get product similarity scores
            scores = hybrid_sim[product_id].copy()
        else:
            # If no product_id, initialize uniform scores
            scores = pd.Series(1.0, index=hybrid_sim.index)
        
        # If user characteristics are provided
        if user_characteristics is not None:
            # Get cached or computed characteristic similarity
            # Create a cache key from the user characteristics
            cache_key = tuple(sorted((k, str(v)) for k, v in user_characteristics.items() 
                               if v is not None and pd.notna(v)))
            
            # Check if result is in cache
            if cache_key in characteristic_cache:
                char_scores = characteristic_cache[cache_key]
            else:
                # Calculate characteristic similarity
                char_scores = calculate_characteristic_similarity(user_characteristics, char_matrix_data)
                # Store in cache
                characteristic_cache[cache_key] = char_scores
            
            # Combine with hybrid scores using beta parameter
            final_scores = (beta * scores) + ((1 - beta) * char_scores)
        else:
            final_scores = scores
        
        # Sort scores
        final_scores = final_scores.sort_values(ascending=False)
        
        # Exclude the queried product if it exists
        if product_id is not None and product_id in final_scores.index:
            final_scores = final_scores[final_scores.index != product_id]
        
        # Get top recommendations
        recommended_product_ids = final_scores.iloc[:num_recommendations].index
        
        # Merge with product info
        recommendations = products_df[products_df["product_id"].isin(recommended_product_ids)][
            ["product_id", "product_name", "brand_name", "price_usd", 
             "primary_category", "secondary_category", "tertiary_category"]
        ].copy()
        
        # Add similarity score
        recommendations["similarity_score"] = [final_scores[pid] for pid in recommendations["product_id"]]
        
        # Sort by similarity score
        recommendations = recommendations.sort_values("similarity_score", ascending=False)
        
        return recommendations
    
    def recommend_for_user_func(author_id, num_recommendations=5, beta=0.5):
        """
        Recommends products for a specific user based on their past ratings and characteristics.
        """
        if author_id not in user_product_matrix.columns:
            raise ValueError(f"No ratings found for user {author_id}")
        
        user_ratings = user_product_matrix[author_id]
        
        # Filter out products the user hasn't rated
        user_ratings = user_ratings[user_ratings > 0]
        
        if len(user_ratings) == 0:
            raise ValueError(f"No ratings found for user {author_id}")
        
        # Get the user's characteristics
        user_chars_df = user_characteristics_df[user_characteristics_df["author_id"] == author_id]
        
        if len(user_chars_df) > 0:
            user_chars = user_chars_df.iloc[0]
            user_characteristics = {
                'skin_tone': user_chars.get('skin_tone'),
                'eye_color': user_chars.get('eye_color'),
                'skin_type': user_chars.get('skin_type'),
                'hair_color': user_chars.get('hair_color')
            }
            
            # Filter out None values
            user_characteristics = {k: v for k, v in user_characteristics.items() 
                                   if v is not None and pd.notna(v)}
        else:
            user_characteristics = {}
        
        # Get products similar to what the user rated highly
        # Weight by the user's rating
        weighted_scores = pd.Series(0.0, index=hybrid_sim.index)
        
        for product_id, rating in user_ratings.items():
            # Normalize rating to 0-1 scale (assuming ratings are 1-5)
            normalized_rating = (rating - 1) / 4
            
            # Get similarity scores and weight by rating
            if product_id in hybrid_sim.index:
                weighted_scores += hybrid_sim[product_id] * normalized_rating
        
        # Normalize scores
        weighted_scores = weighted_scores / len(user_ratings)
        
        # Remove products the user has already rated
        weighted_scores = weighted_scores[~weighted_scores.index.isin(user_ratings.index)]
        
        # Calculate characteristic similarity
        if user_characteristics:
            # Create a cache key from the user characteristics
            cache_key = tuple(sorted((k, str(v)) for k, v in user_characteristics.items() 
                               if v is not None and pd.notna(v)))
            
            # Check if result is in cache
            if cache_key in characteristic_cache:
                char_scores = characteristic_cache[cache_key]
            else:
                # Calculate characteristic similarity
                char_scores = calculate_characteristic_similarity(user_characteristics, char_matrix_data)
                # Store in cache
                characteristic_cache[cache_key] = char_scores
            
            # Combine scores
            final_scores = (beta * weighted_scores) + ((1 - beta) * char_scores)
        else:
            final_scores = weighted_scores
        
        # Get top recommendations
        final_scores = final_scores.sort_values(ascending=False)
        recommended_product_ids = final_scores.iloc[:num_recommendations].index
        
        # Merge with product info
        recommendations = products_df[products_df["product_id"].isin(recommended_product_ids)][
            ["product_id", "product_name", "brand_name", "price_usd", 
             "primary_category", "secondary_category", "tertiary_category"]
        ].copy()
        
        # Add similarity score
        recommendations["similarity_score"] = [final_scores[pid] for pid in recommendations["product_id"]]
        
        # Sort by similarity score
        recommendations = recommendations.sort_values("similarity_score", ascending=False)
        
        return recommendations
    
    def calculate_characteristic_similarity(user_characteristics, char_data):
        """
        Calculate characteristic similarity using matrix operations.
        """
        # Initialize similarity scores
        similarity_scores = pd.Series(0.0, index=char_data['product_ids'])
        
        # Filter valid characteristics
        valid_chars = [k for k, v in user_characteristics.items() 
                     if v is not None and pd.notna(v) 
                     and k in char_data['char_matrices']]
        
        if not valid_chars:
            return similarity_scores
        
        # Calculate match scores for each valid characteristic
        match_scores_list = []
        
        for char in valid_chars:
            char_value = str(user_characteristics[char]).lower()
            char_value_map = char_data['char_value_maps'][char]
            
            if char_value in char_value_map:
                char_idx = char_value_map[char_value]
                
                # Extract the product match vector for this characteristic value
                char_vector = char_data['char_matrices'][char][char_idx].toarray().flatten()
                
                # Calculate the match ratio for each product
                feature_counts = char_data['product_feature_counts'][char]
                
                # Avoid division by zero, if feature count is 0, match ratio is 0
                match_ratios = np.zeros(len(char_data['product_ids']))
                nonzero_indices = feature_counts > 0
                match_ratios[nonzero_indices] = char_vector[nonzero_indices] / feature_counts[nonzero_indices]
                
                match_scores_list.append(match_ratios)
        
        # If no valid matches, return zero scores
        if not match_scores_list:
            return similarity_scores
        
        # Calculate the average match scores for all characteristics
        avg_match_scores = np.mean(match_scores_list, axis=0)
        
        # Convert scores back to Series, using product IDs as index
        for i, score in enumerate(avg_match_scores):
            if score > 0:
                product_id = char_data['product_ids'][i]
                similarity_scores[product_id] = score
        
        return similarity_scores
    
    # Create evaluation object
    evaluator = RecommenderEvaluation(
        products_df=products_df,
        reviews_df=reviews_df,
        user_product_matrix=user_product_matrix,
        hybrid_sim=hybrid_sim,
        char_matrix_data=char_matrix_data,
        user_characteristics_df=user_characteristics_df,
        recommend_products_func=recommend_products_func,
        recommend_for_user_func=recommend_for_user_func
    )
    
    # Run evaluation of key metrics
    metrics = evaluator.evaluate_key_metrics()
    
    # Generate plots
    evaluator.plot_key_metrics()
    
    return evaluator
#%% Evaluate the recommender system
if __name__ == "__main__":
    # Example usage
    evaluator = evaluate_recommender_key_metrics(
        products_csv="products.csv",
        review_data_csv="review_data.csv",
        hybrid_similarity_pkl="hybrid_similarity.pkl",
        char_matrix_data_pkl="char_matrix_data.pkl",
        user_characteristics_csv="user_characteristics.csv"
    )
    
    # Print results
    print("\nKey evaluation metrics:")
    for metric, value in evaluator.metrics.items():
        print(f"{metric}: {value}")
# %%
