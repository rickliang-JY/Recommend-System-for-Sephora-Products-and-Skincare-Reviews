import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(
    page_title="Beauty Product Recommender",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Beauty Product Recommender System")
st.markdown("""
This app recommends beauty products based on your preferences and characteristics.
You can either select a product you like, enter your personal characteristics, or both!
""")

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    return stopwords.words('english'), WordNetLemmatizer()

stop_words, lemmatizer = initialize_nltk()

# Load data
@st.cache_data
def load_data():
    # Load products data
    products_df = pd.read_csv("products.csv")
    
    # Load user characteristics data
    user_characteristics_df = pd.read_csv("user_characteristics.csv")
    
    # Load hybrid similarity matrix
    with open("hybrid_similarity.pkl", "rb") as file:
        hybrid_sim = pickle.load(file)
    
    # Load precomputed characteristic matrices
    with open("char_matrix_data.pkl", "rb") as file:
        char_matrix_data = pickle.load(file)
    
    return products_df, user_characteristics_df, hybrid_sim, char_matrix_data

# Function to calculate characteristic similarity using matrix operations
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

# Initialize cache for characteristic similarity
characteristic_cache = {}

# Recommendation function
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
        else:
            # Calculate characteristic similarity using matrix method
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
    
    return recommendations, elapsed_time

# Load data
try:
    products_df, user_characteristics_df, hybrid_sim, char_matrix_data = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Product-Based", "Characteristic-Based", "Hybrid"])
    
    with tab1:
        st.header("Find Similar Products")
        
        # Create columns for selecting product attributes
        col1, col2 = st.columns(2)
        
        with col1:
            # Get all unique brands and categories
            brands = ["All Brands"] + sorted(products_df["brand_name"].unique().tolist())
            selected_brand = st.selectbox("Select Brand", brands)
        
        with col2:
            # Get primary categories
            categories = ["All Categories"] + sorted(products_df["primary_category"].unique().tolist())
            selected_category = st.selectbox("Select Category", categories)
        
        # Filter products based on selected brand and category
        filtered_products = products_df.copy()
        
        if selected_brand != "All Brands":
            filtered_products = filtered_products[filtered_products["brand_name"] == selected_brand]
        
        if selected_category != "All Categories":
            filtered_products = filtered_products[filtered_products["primary_category"] == selected_category]
        
        # If we have filtered products, display secondary filters
        if not filtered_products.empty:
            col3, col4 = st.columns(2)
            
            with col3:
                # Secondary category filter, only show relevant ones
                if selected_category != "All Categories":
                    sec_categories = ["All"] + sorted(filtered_products["secondary_category"].unique().tolist())
                    selected_sec_category = st.selectbox("Select Sub-Category", sec_categories)
                    
                    if selected_sec_category != "All":
                        filtered_products = filtered_products[filtered_products["secondary_category"] == selected_sec_category]
            
            with col4:
                # Price range filter
                min_price = int(filtered_products["price_usd"].min())
                max_price = int(filtered_products["price_usd"].max() + 1)
                price_range = st.slider("Price Range ($)", min_price, max_price, (min_price, max_price))
                
                filtered_products = filtered_products[
                    (filtered_products["price_usd"] >= price_range[0]) & 
                    (filtered_products["price_usd"] <= price_range[1])
                ]
            
            # Create a selection box for the products
            if not filtered_products.empty:
                st.write(f"Found {len(filtered_products)} products matching your criteria:")
                
                product_options = filtered_products.apply(
                    lambda row: f"{row['brand_name']} - {row['product_name']} (${row['price_usd']:.2f})", axis=1
                ).tolist()
                
                selected_product_str = st.selectbox("Select a product", product_options)
                
                # Get the selected product ID
                selected_idx = product_options.index(selected_product_str)
                selected_product_id = filtered_products.iloc[selected_idx]["product_id"]
                
                # Display product details
                selected_product = products_df[products_df["product_id"] == selected_product_id].iloc[0]
                
                st.subheader("Selected Product")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Product Name:** {selected_product['product_name']}")
                    st.write(f"**Brand:** {selected_product['brand_name']}")
                    st.write(f"**Price:** ${selected_product['price_usd']:.2f}")
                
                with col2:
                    st.write(f"**Category:** {selected_product['primary_category']} > {selected_product['secondary_category']} > {selected_product['tertiary_category']}")
                    rating = selected_product.get('rating', 'N/A')
                    st.write(f"**Rating:** {rating}")
                
                # Get recommendations
                num_recommendations = st.slider("Number of recommendations", 1, 20, 5, key="product_slider")
                
                if st.button("Get Similar Products", key="product_button"):
                    recommendations, elapsed_time = recommend_products(
                        product_id=selected_product_id,
                        num_recommendations=num_recommendations
                    )
                    
                    st.success(f"Found {len(recommendations)} recommendations in {elapsed_time:.4f} seconds!")
                    
                    # Format the recommendations as a table
                    st.dataframe(
                        recommendations[["product_name", "brand_name", "price_usd", "primary_category", "similarity_score"]].rename(
                            columns={
                                "product_name": "Product Name",
                                "brand_name": "Brand",
                                "price_usd": "Price ($)",
                                "primary_category": "Category",
                                "similarity_score": "Similarity Score"
                            }
                        ).set_index("Product Name").style.format({
                            "Similarity Score": "{:.2%}",
                            "Price ($)": "${:.2f}"
                        })
                    )
            else:
                st.warning("No products found matching your search. Try a different query.")
    
    with tab2:
        st.header("Find Products Based on Your Characteristics")
        
        # Get unique skin tones, skin types, hair colors, and eye colors
        skin_tones = sorted(user_characteristics_df["skin_tone"].dropna().unique())
        skin_types = sorted(user_characteristics_df["skin_type"].dropna().unique())
        hair_colors = sorted(user_characteristics_df["hair_color"].dropna().unique())
        eye_colors = sorted(user_characteristics_df["eye_color"].dropna().unique())
        
        # Filter out 'Unknown', 'unknown', etc.
        skin_tones = [s for s in skin_tones if s.lower() != 'unknown']
        skin_types = [s for s in skin_types if s.lower() != 'unknown']
        hair_colors = [s for s in hair_colors if s.lower() != 'unknown']
        eye_colors = [s for s in eye_colors if s.lower() != 'unknown']
        
        # Create columns for the inputs
        col1, col2 = st.columns(2)
        
        with col1:
            selected_skin_tone = st.selectbox("Skin Tone", [""] + skin_tones)
            selected_skin_type = st.selectbox("Skin Type", [""] + skin_types)
        
        with col2:
            selected_hair_color = st.selectbox("Hair Color", [""] + hair_colors)
            selected_eye_color = st.selectbox("Eye Color", [""] + eye_colors)
        
        # Create a dictionary of user characteristics
        user_characteristics = {
            "skin_tone": selected_skin_tone if selected_skin_tone else None,
            "skin_type": selected_skin_type if selected_skin_type else None,
            "hair_color": selected_hair_color if selected_hair_color else None,
            "eye_color": selected_eye_color if selected_eye_color else None
        }
        
        # Remove None values
        user_characteristics = {k: v for k, v in user_characteristics.items() if v is not None}
        
        # Get recommendations
        num_recommendations = st.slider("Number of recommendations", 1, 20, 5, key="char_slider")
        
        if st.button("Get Recommendations", key="char_button"):
            if user_characteristics:
                recommendations, elapsed_time = recommend_products(
                    user_characteristics=user_characteristics,
                    num_recommendations=num_recommendations
                )
                
                st.success(f"Found {len(recommendations)} recommendations in {elapsed_time:.4f} seconds!")
                
                # Format the recommendations as a table
                st.dataframe(
                    recommendations[["product_name", "brand_name", "price_usd", "primary_category", "similarity_score"]].rename(
                        columns={
                            "product_name": "Product Name",
                            "brand_name": "Brand",
                            "price_usd": "Price ($)",
                            "primary_category": "Category",
                            "similarity_score": "Similarity Score"
                        }
                    ).set_index("Product Name").style.format({
                        "Similarity Score": "{:.2%}",
                        "Price ($)": "${:.2f}"
                    })
                )
            else:
                st.warning("Please select at least one characteristic.")
    
    with tab3:
        st.header("Hybrid Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Step 1: Select a Product (Optional)")
            
            # Create dropdown selectors for product selection
            brand_col, category_col = st.columns(2)
            
            with brand_col:
                # Get all unique brands
                brands = ["All Brands"] + sorted(products_df["brand_name"].unique().tolist())
                selected_brand = st.selectbox("Select Brand", brands, key="hybrid_brand")
            
            with category_col:
                # Get primary categories
                categories = ["All Categories"] + sorted(products_df["primary_category"].unique().tolist())
                selected_category = st.selectbox("Select Category", categories, key="hybrid_category")
            
            # Filter products based on selected brand and category
            filtered_products = products_df.copy()
            selected_product_id = None
            
            if selected_brand != "All Brands":
                filtered_products = filtered_products[filtered_products["brand_name"] == selected_brand]
            
            if selected_category != "All Categories":
                filtered_products = filtered_products[filtered_products["primary_category"] == selected_category]
            
            # If we have filtered products, show product selector
            if not filtered_products.empty:
                # Create a selection box for the products
                product_options = filtered_products.apply(
                    lambda row: f"{row['brand_name']} - {row['product_name']} (${row['price_usd']:.2f})", axis=1
                ).tolist()
                
                if product_options:
                    st.write(f"Found {len(product_options)} products matching your criteria:")
                    selected_product_str = st.selectbox("Select a product", product_options, key="hybrid_product")
                    
                    # Get the selected product ID
                    selected_idx = product_options.index(selected_product_str)
                    selected_product_id = filtered_products.iloc[selected_idx]["product_id"]
                    
                    # Display selected product
                    selected_product = products_df[products_df["product_id"] == selected_product_id].iloc[0]
                    st.write(f"Selected: **{selected_product['brand_name']} - {selected_product['product_name']}**")
                else:
                    st.warning("No products found with the selected criteria. Try different filters.")
        
        with col2:
            st.subheader("Step 2: Enter Your Characteristics (Optional)")
            
            # Get unique skin tones, skin types, hair colors, and eye colors
            skin_tones = sorted(user_characteristics_df["skin_tone"].dropna().unique())
            skin_types = sorted(user_characteristics_df["skin_type"].dropna().unique())
            hair_colors = sorted(user_characteristics_df["hair_color"].dropna().unique())
            eye_colors = sorted(user_characteristics_df["eye_color"].dropna().unique())
            
            # Filter out 'Unknown', 'unknown', etc.
            skin_tones = [s for s in skin_tones if s.lower() != 'unknown']
            skin_types = [s for s in skin_types if s.lower() != 'unknown']
            hair_colors = [s for s in hair_colors if s.lower() != 'unknown']
            eye_colors = [s for s in eye_colors if s.lower() != 'unknown']
            
            # Create columns for the inputs
            col1a, col1b = st.columns(2)
            
            with col1a:
                selected_skin_tone = st.selectbox("Skin Tone", [""] + skin_tones, key="hybrid_skin_tone")
                selected_skin_type = st.selectbox("Skin Type", [""] + skin_types, key="hybrid_skin_type")
            
            with col1b:
                selected_hair_color = st.selectbox("Hair Color", [""] + hair_colors, key="hybrid_hair_color")
                selected_eye_color = st.selectbox("Eye Color", [""] + eye_colors, key="hybrid_eye_color")
            
            # Create a dictionary of user characteristics
            user_characteristics = {
                "skin_tone": selected_skin_tone if selected_skin_tone else None,
                "skin_type": selected_skin_type if selected_skin_type else None,
                "hair_color": selected_hair_color if selected_hair_color else None,
                "eye_color": selected_eye_color if selected_eye_color else None
            }
            
            # Remove None values
            user_characteristics = {k: v for k, v in user_characteristics.items() if v is not None}
        
        st.subheader("Step 3: Adjust Parameters")
        
        col3, col4 = st.columns(2)
        
        with col3:
            beta = st.slider(
                "Balance between product similarity and user characteristics", 
                0.0, 1.0, 0.5, 0.1,
                help="0 = Only user characteristics, 1 = Only product similarity"
            )
        
        with col4:
            num_recommendations = st.slider("Number of recommendations", 1, 20, 5, key="hybrid_slider")
        
        # Check if at least one input is provided
        if st.button("Get Hybrid Recommendations", key="hybrid_button"):
            if selected_product_id or user_characteristics:
                recommendations, elapsed_time = recommend_products(
                    product_id=selected_product_id,
                    user_characteristics=user_characteristics,
                    num_recommendations=num_recommendations,
                    beta=beta
                )
                
                st.success(f"Found {len(recommendations)} recommendations in {elapsed_time:.4f} seconds!")
                
                # Format the recommendations as a table
                st.dataframe(
                    recommendations[["product_name", "brand_name", "price_usd", "primary_category", "similarity_score"]].rename(
                        columns={
                            "product_name": "Product Name",
                            "brand_name": "Brand",
                            "price_usd": "Price ($)",
                            "primary_category": "Category",
                            "similarity_score": "Similarity Score"
                        }
                    ).set_index("Product Name").style.format({
                        "Similarity Score": "{:.2%}",
                        "Price ($)": "${:.2f}"
                    })
                )
            else:
                st.warning("Please either select a product or enter your characteristics.")

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Beauty Product Recommender System | Developed with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Please make sure all required data files are available in the same directory as this app.")
    st.markdown("""
    Required files:
    - products.csv
    - user_characteristics.csv
    - hybrid_similarity.pkl
    - char_matrix_data.pkl
    """)