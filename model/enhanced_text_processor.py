import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

class BeautyTextProcessor:
    """
    Specialized text processor for beauty product reviews and descriptions
    with domain-specific stopwords and feature extraction
    """
    
    def __init__(self):
        """Initialize the text processor with domain-specific word sets"""
        # Enhanced stopwords specific to beauty product reviews
        self.enhanced_stopwords = {
            # Basic pronouns and articles
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 
            'those', 'a', 'an', 'the',
                    
            # Common verbs and auxiliary verbs
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 
            'might', 'must', "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
            "doesn't", "don't", "didn't", "won't", "wouldn't", "shouldn't", "can't", "couldn't",
                    
            # Common prepositions
            'on', 'at', 'in', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of', 'off',
            'over', 'under', 'again', 'further', 'then', 'once',
                    
            # Common adverbs and conjunctions
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'not', 'no', 'nor', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
                    
            # Common words in online reviews that don't convey sentiment
            'product', 'products', 'item', 'items', 'bought', 'purchase', 'purchased', 'buy', 'buys',
            'buying', 'using', 'use', 'used', 'uses', 'tried', 'trying', 'try', 'tries', 'reviews', 
            'review', 'reviewed', 'time', 'times', 'day', 'days', 'week', 'weeks', 'month', 'months',
            'year', 'years', 'get', 'gets', 'getting', 'got', 'going', 'went', 'go', 'goes',
            'came', 'come', 'comes', 'coming', 'see', 'sees', 'seeing', 'seen', 'saw',
            'need', 'needs', 'needed', 'want', 'wants', 'wanted', 'amazon', 'online', 'store',
            'shopping', 'shop', 'shopped', 'shops', 'shipping', 'shipped', 'ship', 'ships',
            'order', 'ordered', 'orders', 'ordering', 'arrived', 'arrive', 'arrives', 'arriving',
            'sephora', 'ulta', 'walmart', 'target'
        }
        
        # Common skincare problem indicators
        self.problem_indicators = {
            'dryness': ['dry', 'drying', 'dried', 'dehydrated', 'flaky', 'chapped', 'parched', 'cracked', 'peeling', 'tight'],
            'irritation': ['irritate', 'irritation', 'irritated', 'burn', 'burning', 'sting', 'stinging', 'itch', 'itchy', 'itching', 'red', 'redness', 'inflamed', 'inflammation', 'sensitive'],
            'breakouts': ['breakout', 'pimple', 'acne', 'zit', 'blemish', 'clog', 'clogged', 'blackhead', 'whitehead', 'bump', 'pore', 'pores', 'comedogenic', 'purge', 'purging'],
            'texture_issues': ['sticky', 'greasy', 'oily', 'thick', 'thin', 'watery', 'consistency', 'lumpy', 'gritty', 'grainy', 'rough', 'texture', 'film', 'residue', 'heavy', 'lightweight', 'tacky'],
            'absorption': ['absorb', 'absorption', 'sit', 'surface', 'soak', 'sink', 'penetrate', 'layer', 'pilling'],
            'scent_issues': ['smell', 'scent', 'fragrance', 'odor', 'perfume', 'strong', 'stink', 'stinks', 'stinky', 'cologne', 'aroma', 'artificial'],
            'efficacy': ['ineffective', 'useless', 'waste', 'didn\'t work', 'doesn\'t work', 'no difference', 'no change', 'no improvement', 'no results', 'disappointment', 'disappointed', 'disappointing'],
            'packaging': ['packaging', 'container', 'bottle', 'jar', 'pump', 'dispenser', 'applicator', 'broke', 'broken', 'leaking', 'messy', 'cracked', 'lid', 'cap', 'dropper'],
            'value': ['expensive', 'pricey', 'overpriced', 'cost', 'waste', 'money', 'worth', 'price', 'cheap', 'value', 'affordable', 'budget', 'luxury', 'drugstore', 'splurge'],
            'allergic_reaction': ['allergic', 'allergy', 'reaction', 'hives', 'swelling', 'rash', 'dermatitis', 'eczema'],
            'oxidation': ['oxidized', 'oxidation', 'oxidizing', 'brown', 'yellow', 'turned', 'separated', 'separation']
        }
        
        # Skincare-specific positive descriptors
        self.positive_descriptors = {
            'hydrating', 'moisturizing', 'soothing', 'calming', 'gentle', 'effective', 'refreshing',
            'brightening', 'smoothing', 'firming', 'plumping', 'nourishing', 'healing', 'balancing',
            'softening', 'revitalizing', 'glowing', 'radiant', 'clear', 'clean', 'lightweight',
            'rejuvenating', 'purifying', 'renewing', 'luxurious', 'silky', 'creamy', 'rich',
            'absorbs', 'absorbing', 'absorbed', 'amazing', 'excellent', 'wonderful', 'fantastic',
            'favorite', 'love', 'incredible', 'perfect', 'smooth', 'soft', 'supple', 'fresh',
            'lasting', 'nongreasy', 'affordable', 'worth', 'value', 'helpful', 'recommend'
        }
        
        # Skincare-specific negative descriptors
        self.negative_descriptors = {
            'drying', 'irritating', 'harsh', 'burning', 'stinging', 'itchy', 'greasy', 'sticky',
            'heavy', 'cakey', 'oily', 'thick', 'pilling', 'comedogenic', 'clogging', 'breaking',
            'breakout', 'acne', 'rash', 'reaction', 'allergic', 'redness', 'ineffective', 'useless',
            'waste', 'expensive', 'overpriced', 'disappointing', 'irritated', 'bad', 'worse',
            'worst', 'horrible', 'terrible', 'awful', 'stinks', 'smells', 'strong', 'chemical',
            'artificial', 'messy', 'broken', 'leaking', 'damaged'
        }
        
        # Create lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()
        
        # Create flat list of all problem indicator terms for fast lookup
        self.all_problem_terms = []
        for terms in self.problem_indicators.values():
            self.all_problem_terms.extend(terms)
    
    def clean_text(self, text):
        """
        Clean and preprocess text with domain-specific stopword removal
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        cleaned_tokens = []
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token)
            if lemma not in self.enhanced_stopwords:
                cleaned_tokens.append(lemma)
        
        return ' '.join(cleaned_tokens)
    
    def extract_problem_features(self, text):
        """
        Extract product problem indicators from text
        
        Args:
            text: Cleaned text string
            
        Returns:
            Dictionary with problem category scores
        """
        if not isinstance(text, str):
            return {}
        
        # Convert to lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)
        
        # Lemmatize tokens
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Count problem indicators by category
        problem_scores = {}
        for category, terms in self.problem_indicators.items():
            # Count occurrences of terms in this category
            count = sum(1 for lemma in lemmas if lemma in terms)
            
            # Normalize by text length
            score = count / max(1, len(lemmas))
            problem_scores[f"problem_{category}"] = score
        
        return problem_scores
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment features specific to beauty products
        
        Args:
            text: Cleaned text string
            
        Returns:
            Dictionary with sentiment scores
        """
        if not isinstance(text, str):
            return {}
        
        # Convert to lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)
        
        # Lemmatize tokens
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Count positive and negative terms
        positive_count = sum(1 for lemma in lemmas if lemma in self.positive_descriptors)
        negative_count = sum(1 for lemma in lemmas if lemma in self.negative_descriptors)
        
        # Calculate sentiment scores
        total_terms = max(1, len(lemmas))  # Avoid division by zero
        positive_score = positive_count / total_terms
        negative_score = negative_count / total_terms
        
        # Net sentiment (-1 to 1 scale)
        net_sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'net_sentiment': net_sentiment
        }
    
    def extract_key_terms(self, text, top_n=10):
        """
        Extract important domain-specific terms from text
        
        Args:
            text: Cleaned text string
            top_n: Number of top terms to extract
            
        Returns:
            List of key terms
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)
        
        # Filter out stopwords and lemmatize
        filtered_lemmas = []
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token)
            if lemma not in self.enhanced_stopwords and len(lemma) > 2:
                filtered_lemmas.append(lemma)
        
        # Count term frequencies
        term_counts = Counter(filtered_lemmas)
        
        # Get top N terms
        top_terms = [term for term, count in term_counts.most_common(top_n)]
        
        return top_terms
    
    def process_text(self, text):
        """
        Complete text processing pipeline for beauty product texts
        
        Args:
            text: Raw text string
            
        Returns:
            Dictionary with cleaned text and extracted features
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract features
        problem_features = self.extract_problem_features(cleaned_text)
        sentiment_features = self.extract_sentiment_features(cleaned_text)
        key_terms = self.extract_key_terms(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'problem_features': problem_features,
            'sentiment_features': sentiment_features,
            'key_terms': key_terms
        }


# Function to integrate the text processor with the dual tower model
def integrate_with_dual_tower(text_processor, review_text, product_text):
    """
    Process review and product text for the dual tower model
    
    Args:
        text_processor: BeautyTextProcessor instance
        review_text: Raw review text
        product_text: Raw product text (name + description)
        
    Returns:
        Processed features for both towers
    """
    # Process review text
    review_processed = text_processor.process_text(review_text)
    
    # Process product text
    product_processed = text_processor.process_text(product_text)
    
    # Extract features for model input
    review_features = {
        'text': review_processed['cleaned_text'],
        **review_processed['problem_features'],
        **review_processed['sentiment_features']
    }
    
    product_features = {
        'text': product_processed['cleaned_text'],
        **product_processed['problem_features']
    }
    
    return review_features, product_features


# Example usage
if __name__ == "__main__":
    # Initialize text processor
    processor = BeautyTextProcessor()
    
    # Example review
    example_review = """
    I've been using this moisturizer for about 3 weeks now and I'm really impressed! 
    My skin is usually very dry and flaky, especially in the winter, but this cream has 
    been keeping it hydrated and smooth. It absorbs quickly without feeling greasy and 
    doesn't break me out like some heavy moisturizers do. The scent is very mild and pleasant. 
    A little goes a long way so the jar should last a while. Definitely worth the price!
    """
    
    # Example product description
    example_product = """
    Ultimate Hydrating Moisturizer: This luxurious cream provides 24-hour hydration for 
    all skin types. Enriched with hyaluronic acid, ceramides, and vitamin E to restore 
    the skin's moisture barrier. Fragrance-free formula is gentle enough for sensitive skin. 
    Helps reduce the appearance of fine lines and gives skin a plump, dewy finish.
    """
    
    # Process the texts
    processed_review = processor.process_text(example_review)
    processed_product = processor.process_text(example_product)
    
    # Print results
    print("CLEANED REVIEW TEXT:")
    print(processed_review['cleaned_text'])
    print("\nREVIEW PROBLEM FEATURES:")
    print(processed_review['problem_features'])
    print("\nREVIEW SENTIMENT:")
    print(processed_review['sentiment_features'])
    print("\nREVIEW KEY TERMS:")
    print(processed_review['key_terms'])
    
    print("\n" + "="*50 + "\n")
    
    print("CLEANED PRODUCT TEXT:")
    print(processed_product['cleaned_text'])
    print("\nPRODUCT KEY TERMS:")
    print(processed_product['key_terms'])