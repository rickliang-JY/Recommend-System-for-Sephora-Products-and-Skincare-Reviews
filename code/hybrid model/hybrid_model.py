#%% Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import time
from collections import Counter
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%% Functions
def remove_non_english(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Advanced text cleaning function from CBF_model.py
def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def precompute_characteristic_matrix(user_characteristics_df, products_df):
    """
    Precompute characteristic matrices for fast similarity calculation
    
    Parameters:
    -----------
    user_characteristics_df : pandas DataFrame
        DataFrame containing user characteristic information
    products_df : pandas DataFrame
        DataFrame containing product information
        
    Returns:
    --------
    dict of characteristic matrices and mappings
    """
    print("Precomputing characteristic matrices...")
    start_time = time.time()
    
    # List of characteristics
    chars = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
    
    # Get all unique product IDs and author IDs
    product_ids = products_df['product_id'].unique()
    author_ids = user_characteristics_df['author_id'].unique()
    
    # Create a mapping from product ID to index
    product_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    
    # Create dictionaries for each characteristic to map values to one-hot encoding indices
    char_value_maps = {}
    char_matrices = {}
    
    # Create a product feature count matrix to track the number of reviews for each product for each characteristic
    product_feature_counts = {}
    
    # Create matrices for each characteristic
    for char in chars:
        if char in user_characteristics_df.columns:
            # Get all unique non-null values for the characteristic
            unique_values = user_characteristics_df[char].dropna().unique()
            
            # Create a mapping from value to index
            value_to_idx = {val.lower(): i for i, val in enumerate(unique_values) if pd.notna(val)}
            char_value_maps[char] = value_to_idx
            
            # Create a characteristic-product matrix (number of characteristic values x number of products)
            matrix = np.zeros((len(value_to_idx), len(product_ids)))
            
            # Initialize the product feature count dictionary for this characteristic
            product_feature_counts[char] = np.zeros(len(product_ids))
            
            # Iterate over each product-characteristic combination and fill the matrix
            for _, row in user_characteristics_df.dropna(subset=[char]).iterrows():
                if pd.notna(row[char]) and row['product_id'] in product_to_idx:
                    char_value = row[char].lower()
                    if char_value in value_to_idx:
                        char_idx = value_to_idx[char_value]
                        prod_idx = product_to_idx[row['product_id']]
                        matrix[char_idx, prod_idx] += 1.0  # Increment the count of this characteristic value
                        product_feature_counts[char][prod_idx] += 1.0  # Increment the count of reviews for this product for this characteristic
            
            # Convert the matrix to a sparse matrix to save memory
            char_matrices[char] = csr_matrix(matrix)
    
    # Create a result dictionary with product IDs and mappings
    result = {
        'product_ids': product_ids,
        'product_to_idx': product_to_idx,
        'char_value_maps': char_value_maps,
        'char_matrices': char_matrices,
        'product_feature_counts': product_feature_counts
    }
    
    elapsed_time = time.time() - start_time
    print(f"Characteristic matrices precomputed in {elapsed_time:.2f} seconds")
    
    return result

def calculate_characteristic_similarity_matrix(user_characteristics, char_data):
    """
    Calculate characteristic similarity using matrix operations
    
    Parameters:
    -----------
    user_characteristics : dict
        Dictionary containing user characteristics such as 'skin_tone', 'eye_color', 'skin_type', 'hair_color'
    char_data : dict
        Precomputed characteristic matrix data
        
    Returns:
    --------
    pd.Series with similarity scores for each product
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

def recommend_products(product_id=None, user_characteristics=None, num_recommendations=5, alpha=0.7, beta=0.5):
    """
    Recommends products based on a hybrid approach combining:
    1. Content-based filtering
    2. Collaborative filtering
    3. User characteristics matching (if provided)
    
    Parameters:
    -----------
    product_id : str, optional
        The ID of the product to base recommendations on
    user_characteristics : dict, optional
        Dictionary containing user characteristics such as skin_tone, eye_color, skin_type, hair_color
    num_recommendations : int, default=5
        Number of recommendations to return
    alpha : float, default=0.7
        Weight for content-based vs. collaborative filtering (higher means more content-based)
    beta : float, default=0.5
        Weight for hybrid product similarity vs. user characteristic matching (higher means more product similarity)
        
    Returns:
    --------
    DataFrame containing recommended products with their details
    """
    start_time = time.time()
    
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
        global characteristic_cache
        
        # Create a cache key from the user characteristics
        cache_key = tuple(sorted((k, str(v)) for k, v in user_characteristics.items() 
                           if v is not None and pd.notna(v)))
        
        # Check if result is in cache
        if cache_key in characteristic_cache:
            char_scores = characteristic_cache[cache_key]
            print(f"Using cached characteristic scores for {cache_key}")
        else:
            # Calculate characteristic similarity using matrix method
            global char_matrix_data
            char_scores = calculate_characteristic_similarity_matrix(user_characteristics, char_matrix_data)
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
    
    elapsed_time = time.time() - start_time
    print(f"Recommendation completed in {elapsed_time:.4f} seconds")
    
    return recommendations

# def recommend_products(
#     product_id=None, 
#     user_characteristics=None, 
#     user_id=None,
#     num_recommendations=5, 
#     content_weight=0.7, 
#     beta=0.5,
#     exclude_viewed=True,
#     min_similarity=0.05,
#     filter_criteria=None
# ):
#     """
#     Enhanced recommendation function combining multiple recommendation strategies.
    
#     Parameters:
#     -----------
#     product_id : str, optional
#         The ID of the product to base recommendations on
#     user_characteristics : dict, optional
#         Dictionary containing user characteristics such as skin_tone, eye_color, skin_type, hair_color
#     user_id : str, optional
#         User ID to personalize recommendations based on past ratings (collaborative filtering)
#     num_recommendations : int, default=5
#         Number of recommendations to return
#     content_weight : float, default=0.7
#         Weight for content-based vs. collaborative filtering (higher means more content-based)
#     beta : float, default=0.5
#         Weight for product similarity vs. user characteristic matching (higher means more product similarity)
#     exclude_viewed : bool, default=True
#         Whether to exclude products the user has already viewed/rated (only applies when user_id is provided)
#     min_similarity : float, default=0.1
#         Minimum similarity score threshold for recommendations
#     filter_criteria : dict, optional
#         Dictionary of filtering criteria, e.g., {'price_usd': {'min': 10, 'max': 50}, 'primary_category': ['Makeup']}
        
#     Returns:
#     --------
#     DataFrame containing recommended products with their details and similarity scores
#     """
#     start_time = time.time()
    
#     if product_id is None and user_characteristics is None and user_id is None:
#         raise ValueError("At least one of product_id, user_characteristics, or user_id must be provided")
    
#     # Initialize final scores
#     final_scores = pd.Series(0.0, index=hybrid_sim.index)
    
#     # 1. Content-based + Collaborative hybrid recommendations (if product_id is provided)
#     if product_id is not None:
#         if product_id not in hybrid_sim.index:
#             raise ValueError(f"Product ID {product_id} not found in the similarity matrix")
#         product_scores = hybrid_sim[product_id].copy()
#         final_scores += product_scores
    
#     # 2. User-based recommendations (if user_id is provided)
#     user_rated_products = []
#     if user_id is not None:
#         if user_id not in user_product_matrix.columns:
#             print(f"Warning: No ratings found for user {user_id}")
#         else:
#             # Get user's ratings
#             user_ratings = user_product_matrix[user_id]
#             user_rated_products = user_ratings[user_ratings > 0].index.tolist()
            
#             # Weight by normalized rating (assuming 1-5 scale)
#             weighted_scores = pd.Series(0.0, index=hybrid_sim.index)
            
#             for pid, rating in user_ratings[user_ratings > 0].items():
#                 # Normalize rating to 0-1 scale
#                 normalized_rating = (rating - 1) / 4
                
#                 # Get similarity scores and weight by rating
#                 if pid in hybrid_sim.index:
#                     weighted_scores += hybrid_sim[pid] * normalized_rating
            
#             # If user has rated products, normalize and add to final scores
#             if len(user_rated_products) > 0:
#                 weighted_scores = weighted_scores / len(user_rated_products)
#                 final_scores += weighted_scores
    
#     # 3. Characteristic-based recommendations (if user_characteristics is provided)
#     if user_characteristics is not None:
#         # Create a cache key from valid characteristics
#         valid_chars = {k: v for k, v in user_characteristics.items() 
#                       if v is not None and pd.notna(v)}
        
#         if valid_chars:
#             cache_key = tuple(sorted((k, str(v)) for k, v in valid_chars.items()))
            
#             # Check if result is in cache
#             if cache_key in characteristic_cache:
#                 char_scores = characteristic_cache[cache_key]
#                 print(f"Using cached characteristic scores")
#             else:
#                 # Calculate characteristic similarity
#                 char_scores = calculate_characteristic_similarity_matrix(valid_chars, char_matrix_data)
#                 # Store in cache
#                 characteristic_cache[cache_key] = char_scores
            
#             # Apply characteristic weight
#             final_scores = (beta * final_scores) + ((1 - beta) * char_scores)
#         else:
#             print("Warning: No valid user characteristics provided")
    
#     # Optional: Exclude already viewed/rated products
#     if exclude_viewed and user_id is not None and user_rated_products:
#         final_scores = final_scores[~final_scores.index.isin(user_rated_products)]
    
#     # Optional: Exclude the queried product
#     if product_id is not None and product_id in final_scores.index:
#         final_scores = final_scores[final_scores.index != product_id]
    
#     # Apply minimum similarity threshold
#     final_scores = final_scores[final_scores >= min_similarity]
    
#     # Sort scores
#     final_scores = final_scores.sort_values(ascending=False)
    
#     # Get product info for recommended products
#     recommended_product_ids = final_scores.iloc[:num_recommendations*2].index  # Get extra for filtering
#     recommendations_df = products_df[products_df["product_id"].isin(recommended_product_ids)].copy()
    
#     # Apply filtering criteria if provided
#     if filter_criteria is not None:
#         for column, criteria in filter_criteria.items():
#             if column in recommendations_df.columns:
#                 if isinstance(criteria, dict):
#                     # Range filter (min/max)
#                     if 'min' in criteria:
#                         recommendations_df = recommendations_df[recommendations_df[column] >= criteria['min']]
#                     if 'max' in criteria:
#                         recommendations_df = recommendations_df[recommendations_df[column] <= criteria['max']]
#                 elif isinstance(criteria, list):
#                     # List of allowed values
#                     recommendations_df = recommendations_df[recommendations_df[column].isin(criteria)]
    
#     # Add similarity scores
#     recommendations_df["similarity_score"] = recommendations_df["product_id"].map(final_scores)
    
#     # Sort by similarity score and take top N
#     recommendations_df = recommendations_df.sort_values("similarity_score", ascending=False)
#     recommendations_df = recommendations_df.head(num_recommendations)
    
#     # Select columns for display
#     display_columns = [
#         "product_id", "product_name", "brand_name", "price_usd", 
#         "primary_category", "secondary_category", "tertiary_category", 
#         "similarity_score"
#     ]
    
#     # Keep only columns that exist
#     valid_columns = [col for col in display_columns if col in recommendations_df.columns]
#     recommendations_df = recommendations_df[valid_columns].copy()
    
#     elapsed_time = time.time() - start_time
#     print(f"Recommendation completed in {elapsed_time:.4f} seconds")
    
#     return recommendations_df

def recommend_for_user(author_id, num_recommendations=5, alpha=0.7, beta=0.5):
    """
    Recommends products for a specific user based on:
    1. Their past ratings (collaborative filtering)
    2. Products similar to what they've rated highly (content-based)
    3. Their personal characteristics
    
    Parameters:
    -----------
    author_id : str
        The ID of the user to recommend products for
    num_recommendations : int, default=5
        Number of recommendations to return
    alpha : float, default=0.7
        Weight for content-based vs. collaborative filtering
    beta : float, default=0.5
        Weight for product similarity vs. user characteristic matching
        
    Returns:
    --------
    DataFrame containing recommended products with their details
    """
    start_time = time.time()
    
    # Get the user's ratings
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
        global characteristic_cache
        if cache_key in characteristic_cache:
            char_scores = characteristic_cache[cache_key]
            print(f"Using cached characteristic scores for {cache_key}")
        else:
            # Calculate characteristic similarity using matrix method
            global char_matrix_data
            char_scores = calculate_characteristic_similarity_matrix(user_characteristics, char_matrix_data)
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
    
    elapsed_time = time.time() - start_time
    print(f"User recommendation completed in {elapsed_time:.4f} seconds")
    
    return recommendations

def prepare_user_characteristics_data(reviews_df):
    """
    Prepare user characteristics data from the reviews dataframe
    Assumes that reviews_df already contains columns for skin_tone, eye_color, skin_type, and hair_color
    """
    # Select relevant columns
    characteristics_columns = ['author_id', 'product_id', 'skin_tone', 'eye_color', 'skin_type', 'hair_color']
    
    # Check which columns actually exist in the dataframe
    existing_columns = [col for col in characteristics_columns if col in reviews_df.columns]
    
    # Create a user characteristics dataframe with the existing columns
    user_characteristics_df = reviews_df[existing_columns].copy()
    
    # Add any missing columns as None
    for col in characteristics_columns:
        if col not in existing_columns:
            user_characteristics_df[col] = None
    
    # Drop duplicates to keep the dataset clean
    user_characteristics_df = user_characteristics_df.drop_duplicates()
    
    return user_characteristics_df
# Text cleaning function
def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Simple tokenization by splitting on whitespace instead of using nltk.word_tokenize
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

#%% Load Data
print("Loading product and review data...")
products_df = pd.read_csv("product_info.csv", encoding='ISO-8859-1')

# products_df = products_df[["product_id", "product_name", "brand_name", "ingredients", "highlights",
#                            "price_usd", "primary_category", "secondary_category", "tertiary_category"]]


reviews_df = pd.read_csv("review_data.csv", encoding='ISO-8859-1')

#%% Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#%% Data Clean
print("Cleaning data...")
reviews_df['review_text'] = reviews_df['review_text'].apply(lambda x: remove_non_english(x) if isinstance(x, str) else x)
reviews_df['review_title'] = reviews_df['review_title'].apply(lambda x: remove_non_english(x) if isinstance(x, str) else x)
reviews_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

reviews_df["rating"] = pd.to_numeric(reviews_df["rating"], errors="coerce")
reviews_df['rating'] = reviews_df['rating'].fillna(0)
reviews_df['is_recommended'] = reviews_df['is_recommended'].fillna(0).astype(int)
reviews_df['skin_type'] = reviews_df['skin_type'].fillna('Unknown')
reviews_df['skin_tone'] = reviews_df['skin_tone'].fillna('Unknown')
reviews_df['hair_color'] = reviews_df['hair_color'].fillna('Unknown')
reviews_df['eye_color'] = reviews_df['eye_color'].fillna('Unknown')
reviews_df['review_text'] = reviews_df['review_text'].fillna('')

# Clean review text
reviews_df['review_text_cleaned'] = reviews_df['review_text'].apply(clean_text)


# Convert submission time to datetime if it's not already
if 'submission_time' in reviews_df.columns:
    reviews_df['submission_time'] = pd.to_datetime(reviews_df['submission_time'], errors='coerce')


# Create user characteristics dataframe
print("Preparing user characteristics data...")
user_characteristics_df = prepare_user_characteristics_data(reviews_df)

# Save it for future use
user_characteristics_df.to_csv("user_characteristics.csv", index=False)

# Initialize global cache for characteristic similarity
characteristic_cache = {}

# Precompute characteristic matrices for fast lookups
char_matrix_data = precompute_characteristic_matrix(user_characteristics_df, products_df)


#%% Preprocessing
# Handle missing values

# Product Name - Fill with empty string (no NA values)
products_df['product_name'] = products_df['product_name'].fillna('')

# Brand Name - Fill with 'Unknown' (no NA values)
products_df['brand_name'] = products_df['brand_name'].fillna('Unknown')

# Ingredients and Highlights - Fill with empty strings
products_df['ingredients'] = products_df['ingredients'].fillna('')
products_df['highlights'] = products_df['highlights'].fillna('')

# Categories - Fill with 'Unknown' (handle NA values)
products_df['primary_category'] = products_df['primary_category'].fillna('Unknown')
products_df['secondary_category'] = products_df['secondary_category'].fillna('Unknown')
products_df['tertiary_category'] = products_df['tertiary_category'].fillna('Unknown')

# Impute Price with Median Price Grouped by Tertiary Category
products_df['price_usd'] = products_df.groupby('tertiary_category')['price_usd'].transform(
    lambda x: x.fillna(x.median())
)

# Impute Rating with Median Rating Grouped by Tertiary Category
products_df['rating'] = products_df.groupby('tertiary_category')['rating'].transform(
    lambda x: x.fillna(x.median())
)

# Drop the row where rating is 315
products_df = products_df[products_df['rating'] != 315]

# Impute the missing value in 'sephora_exclusive' with 0
products_df['sephora_exclusive'] = products_df['sephora_exclusive'].fillna(0)

# Replace any values that are not '0' or '1' with 0
products_df['sephora_exclusive'] = products_df['sephora_exclusive'].apply(
    lambda x: 0 if str(x) not in ['0', '1'] else x
)


# Standardize boolean columns
bool_columns = ['limited_edition', 'new', 'online_only', 'out_of_stock', 'sephora_exclusive']
for col in bool_columns:
    products_df[col] = products_df[col].astype(int)


# Text cleaning function
def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Simple tokenization by splitting on whitespace instead of using nltk.word_tokenize
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)
    
# Clean text columns
print("Cleaning text data...")
products_df['ingredients_cleaned'] = products_df['ingredients'].apply(clean_text)
products_df['highlights_cleaned'] = products_df['highlights'].apply(clean_text)
products_df['product_name_cleaned'] = products_df['product_name'].apply(lambda x: clean_text(str(x)))



# Combine text features
# This combines all relevant textual information about the product into one field for feature extraction or vectorization.
products_df['combined_text'] = (
    products_df['product_name_cleaned'] + ' ' + 
    products_df['ingredients_cleaned'] + ' ' + 
    products_df['highlights_cleaned']
)


# Scale numerical features
scaler = MinMaxScaler()
products_df['price_scaled'] = scaler.fit_transform(products_df[['price_usd']])
products_df['rating_scaled'] = products_df['rating'] / 5.0  # Assuming 5-star rating system



#%% Preprocessing Reviews Data
# Handle missing values
reviews_df['rating'] = reviews_df['rating'].fillna(0)
reviews_df['is_recommended'] = reviews_df['is_recommended'].fillna(0).astype(int)
reviews_df['skin_type'] = reviews_df['skin_type'].fillna('Unknown')
reviews_df['skin_tone'] = reviews_df['skin_tone'].fillna('Unknown')
reviews_df['hair_color'] = reviews_df['hair_color'].fillna('Unknown')
reviews_df['eye_color'] = reviews_df['eye_color'].fillna('Unknown')
reviews_df['review_text'] = reviews_df['review_text'].fillna('')


# Clean review text
reviews_df['review_text_cleaned'] = reviews_df['review_text'].apply(clean_text)


# Convert submission time to datetime if it's not already
if 'submission_time' in reviews_df.columns:
    reviews_df['submission_time'] = pd.to_datetime(reviews_df['submission_time'], errors='coerce')


# Aggregate reviews by product
print("Aggregating reviews by product...")
review_aggs = reviews_df.groupby('product_id').agg({
    'rating': 'mean',
    'is_recommended': 'mean',
    'total_feedback_count': 'sum',
    'total_pos_feedback_count': 'sum',
    'total_neg_feedback_count': 'sum',
    'review_text_cleaned': lambda x: ' '.join(x)
})

# Rename aggregated columns
review_aggs.columns = [
    'user_rating_avg',
    'recommendation_rate',
    'total_feedback',
    'total_positive_feedback',
    'total_negative_feedback',
    'all_reviews_text'
]

# Calculate review count per product
product_review_counts = reviews_df['product_id'].value_counts().to_frame()
product_review_counts.columns = ['review_count']
review_aggs = review_aggs.join(product_review_counts)


# Calculate weighted rating (Wilson score lower bound for 95% confidence)
# To reduce bias and improve the reliability of product ratings, especially in scenarios where some products have very few reviews.
def wilson_score(pos, n):
    if n == 0:
        return 0
    z = 1.96  # 95% confidence
    phat = pos / n
    return (phat + z*z/(2*n) - z*np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)

review_aggs['wilson_score'] = review_aggs.apply(
    lambda x: wilson_score(x['recommendation_rate'] * x['review_count'], x['review_count']), 
    axis=1
)


#%% User Demographic Features
# Extract user demographic preferences per product
# Skin type preferences
skin_type_prefs = pd.crosstab(
    reviews_df['product_id'], 
    reviews_df['skin_type'], 
    normalize='index'
).add_prefix('skin_type_')

# Skin tone preferences
skin_tone_prefs = pd.crosstab(
    reviews_df['product_id'], 
    reviews_df['skin_tone'], 
    normalize='index'
).add_prefix('skin_tone_')

# Hair color preferences
hair_color_prefs = pd.crosstab(
    reviews_df['product_id'], 
    reviews_df['hair_color'], 
    normalize='index'
).add_prefix('hair_color_')

# Eye color preferences
eye_color_prefs = pd.crosstab(
    reviews_df['product_id'], 
    reviews_df['eye_color'], 
    normalize='index'
).add_prefix('eye_color_')

# Merge all demographic preferences
demographic_features = pd.concat(
    [skin_type_prefs, skin_tone_prefs, hair_color_prefs, eye_color_prefs], 
    axis=1
).fillna(0)

demographic_features.head()


demographic_features.iloc[10]


#%% Merged Dataset
# Merge product data with review aggregates
merged_data = pd.merge(
    products_df,
    review_aggs,
    left_on='product_id',
    right_index=True,
    how='left'
)

# Fill missing review data with defaults
merged_data['user_rating_avg'] = merged_data['user_rating_avg'].fillna(merged_data['rating'])
merged_data['recommendation_rate'] = merged_data['recommendation_rate'].fillna(0.5)
merged_data['review_count'] = merged_data['review_count'].fillna(0)
merged_data['wilson_score'] = merged_data['wilson_score'].fillna(0)
merged_data['all_reviews_text'] = merged_data['all_reviews_text'].fillna('')

# Add demographic features
merged_data = pd.merge(
    merged_data,
    demographic_features,
    left_on='product_id',
    right_index=True,
    how='left'
).fillna(0)

merged_data.head()


#%% Feature Engineering
# 1. Handle primary_category (11 unique values)
# Standard one-hot encoding is fine for this small number
primary_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
primary_encoded = primary_encoder.fit_transform(products_df[['primary_category']])
primary_feature_names = [f'primary_{cat}' for cat in primary_encoder.categories_[0]]
primary_df = pd.DataFrame(
    primary_encoded,
    columns=primary_feature_names,
    index=products_df.index
)

# 2. Handle secondary_category (43 unique values)
# Standard one-hot encoding is still reasonable here
secondary_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
secondary_encoded = secondary_encoder.fit_transform(products_df[['secondary_category']])
secondary_feature_names = [f'secondary_{cat}' for cat in secondary_encoder.categories_[0]]
secondary_df = pd.DataFrame(
    secondary_encoded,
    columns=secondary_feature_names,
    index=products_df.index
)

# 3. Handle tertiary_category (119 unique values)
# Group less frequent categories into "Other"
tertiary_counts = products_df['tertiary_category'].value_counts()
top_tertiary = tertiary_counts[tertiary_counts >= 10].index  # Keep categories with at least 10 products
products_df['tertiary_category_grouped'] = products_df['tertiary_category'].apply(
    lambda x: x if x in top_tertiary else 'Other'
)

# One-hot encode the grouped tertiary categories
tertiary_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
tertiary_encoded = tertiary_encoder.fit_transform(products_df[['tertiary_category_grouped']])
tertiary_feature_names = [f'tertiary_{cat}' for cat in tertiary_encoder.categories_[0]]
tertiary_df = pd.DataFrame(
    tertiary_encoded,
    columns=tertiary_feature_names,
    index=products_df.index
)

# 4. Handle brand_name (305 unique values)
# Keep only top brands, group others as "Other"
brand_counts = products_df['brand_name'].value_counts()
top_brands = brand_counts[brand_counts >= 10].index  # Keep brands with at least 10 products
products_df['brand_name_grouped'] = products_df['brand_name'].apply(
    lambda x: x if x in top_brands else 'Other'
)

# One-hot encode the grouped brands
brand_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
brand_encoded = brand_encoder.fit_transform(products_df[['brand_name_grouped']])
brand_feature_names = [f'brand_{brand}' for brand in brand_encoder.categories_[0]]
brand_df = pd.DataFrame(
    brand_encoded,
    columns=brand_feature_names,
    index=products_df.index
)

# Print category reduction summary
print(f"Primary categories: {len(primary_encoder.categories_[0])} (original: 11)")
print(f"Secondary categories: {len(secondary_encoder.categories_[0])} (original: 43)")
print(f"Tertiary categories: {len(tertiary_encoder.categories_[0])} (original: 119, reduced)")
print(f"Brands: {len(brand_encoder.categories_[0])} (original: 305, reduced)")



# 1. TF-IDF for text data
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(merged_data['combined_text'])
tfidf_feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_feature_names,
    index=merged_data.index
)

# 2. Numerical features
numerical_features = merged_data[[
    'price_scaled', 
    'rating_scaled', 
    'user_rating_avg', 
    'recommendation_rate',
    'wilson_score',
    'review_count'
]].copy()

# Scale review count (log transformation to handle skewness)
numerical_features['review_count_scaled'] = np.log1p(numerical_features['review_count'])
numerical_features = numerical_features.fillna(0)

# 3. Boolean features
boolean_features = merged_data[bool_columns].copy()


# Combine all feature matrices
print("Combining all features...")
all_features = [
    tfidf_df,              # Text features
    brand_df,              # Brand features
    primary_df,            # Primary category
    secondary_df,          # Secondary category
    tertiary_df,           # Tertiary category
    numerical_features,    # Numerical features
    boolean_features       # Boolean features
]

# Add demographic features if they exist
if not demographic_features.empty:
    # Reindex to match merged_data
    demo_df = demographic_features.reindex(merged_data.index).fillna(0)
    all_features.append(demo_df)

# Concatenate all features
feature_matrix = pd.concat(all_features, axis=1)

# Handle any NaN values in the feature matrix (critical for cosine_similarity)
feature_matrix = feature_matrix.fillna(0)
print(f"Final CBF feature matrix shape: {feature_matrix.shape}")

# Check for any remaining NaN values
if feature_matrix.isna().any().any():
    print("WARNING: Feature matrix still contains NaN values. Filling with zeros...")
    feature_matrix = feature_matrix.fillna(0)

# Compute improved cosine similarity
print("Computing CBF similarity matrix...")
cosine_sim = cosine_similarity(feature_matrix)

# Convert similarity matrix to DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=products_df["product_id"], columns=products_df["product_id"])

print("Advanced content-based filtering model built successfully!")

#%% Collaborative Filtering (CF)
print("Building collaborative filtering model...")
# Aggregate by taking the mean rating per (product_id, author_id)
reviews_agg = reviews_df.groupby(["product_id", "author_id"], as_index=False).agg({"rating": "mean"})

# Now pivot the table
user_product_matrix = reviews_agg.pivot(index="product_id", columns="author_id", values="rating")

# Fill missing values with 0 (unrated products)
user_product_matrix = user_product_matrix.fillna(0)

# Compute cosine similarity between products based on user ratings
item_similarity = cosine_similarity(user_product_matrix)

# Convert to DataFrame
item_sim_df = pd.DataFrame(item_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)


#%% Hybrid Recommender – Combining Both Models
print("Building hybrid recommendation model...")
# Find missing products
missing_products = list(set(cosine_sim_df.index) - set(item_sim_df.index))

# Create a DataFrame with zeros for missing products
missing_sim_matrix = pd.DataFrame(0, index=missing_products, columns=item_sim_df.columns)

# Append to the collaborative filtering similarity matrix
item_sim_df = pd.concat([item_sim_df, missing_sim_matrix])

# Now add missing products as columns (ensuring symmetry)
missing_sim_matrix = pd.DataFrame(0, index=item_sim_df.index, columns=missing_products)
item_sim_df = pd.concat([item_sim_df, missing_sim_matrix], axis=1)

# Sort rows and columns to match content-based filtering
item_sim_df = item_sim_df.loc[cosine_sim_df.index, cosine_sim_df.index]

# Get product IDs in each matrix
content_products = set(cosine_sim_df.index)
cf_products = set(item_sim_df.index)

# Find missing products in each matrix
missing_in_content = cf_products - content_products
missing_in_cf = content_products - cf_products

print(f"Products in CF but missing in Content-Based: {len(missing_in_content)}")
print(f"Products in Content-Based but missing in CF: {len(missing_in_cf)}")

# Create hybrid similarity matrix
alpha = 0.7  # Weight for Content-Based Filtering (higher means more importance)
hybrid_sim = (alpha * cosine_sim_df) + ((1 - alpha) * item_sim_df)

#%% Save the Trained Models
print("Saving trained models...")
# Save the hybrid similarity matrix
with open("hybrid_similarity.pkl", "wb") as file:
    pickle.dump(hybrid_sim, file)

# Save product metadata for lookup
products_df.to_csv("products.csv", index=False)

# Save user characteristic data
user_characteristics_df.to_csv("user_characteristics.csv", index=False)

# Save precomputed characteristic matrices
with open("char_matrix_data.pkl", "wb") as file:
    pickle.dump(char_matrix_data, file)

# Save feature engineering components for future use
# with open("cbf_model_components.pkl", "wb") as file:
#     pickle.dump({
#         'tfidf_vectorizer': tfidf,
#         'primary_encoder': primary_encoder,
#         'secondary_encoder': secondary_encoder,
#         'brand_encoder': brand_encoder,
#         'tertiary_encoder': 
#         'scaler': scaler
#     }, file)

with open("cbf_model_components.pkl", "wb") as file:
    pickle.dump({
        'tfidf_vectorizer': tfidf,
        'primary_encoder': primary_encoder,
        'secondary_encoder': secondary_encoder,
        'tertiary_encoder': tertiary_encoder,
        'brand_encoder': brand_encoder,
        'feature_names': feature_matrix.columns.tolist()
    }, file)



print("Model training and optimization complete!")

#%% Example Usage

# Example 1: Recommend based on product ID
print("\nExample 1: Recommend based on product ID")
product_id = products_df["product_id"].iloc[0]  # Example product ID
recommendations = recommend_products(product_id=product_id, num_recommendations=5)
print(recommendations[["product_name", "brand_name", "similarity_score"]])

# Example 2: Recommend based on user characteristics
print("\nExample 2: Recommend based on user characteristics")
user_characteristics = {
    "skin_tone": "medium",
    "skin_type": "combination",
    "eye_color": "brown"
}
recommendations = recommend_products(user_characteristics=user_characteristics, num_recommendations=5)
print(recommendations[["product_name", "brand_name", "similarity_score"]])

# Example 3: Recommend based on both product ID and user characteristics
print("\nExample 3: Recommend based on both product ID and user characteristics")
recommendations = recommend_products(
    product_id=product_id,
    user_characteristics=user_characteristics,
    num_recommendations=5,
    beta=0.3  # Give more weight to product similarity
)
print(recommendations[["product_name", "brand_name", "similarity_score"]])

# Example 4: Recommend for a specific user by author_id
print("\nExample 4: Recommend for a specific user by author_id")
try:
    # Try with the first author ID in the dataset
    author_id = user_product_matrix.columns[0]
    recommendations = recommend_for_user(author_id=author_id, num_recommendations=5)
    print(recommendations[["product_name", "brand_name", "similarity_score"]])
except ValueError as e:
    print(f"Error: {e}")

# Example 5: Repeated query with same characteristics (should use cache)
print("\nExample 5: Repeated query with same characteristics (should use cache)")
start_time = time.time()
recommendations = recommend_products(user_characteristics=user_characteristics, num_recommendations=5)
print(f"Total time including output: {time.time() - start_time:.4f} seconds")
print(recommendations[["product_name", "brand_name", "similarity_score"]])
# %%