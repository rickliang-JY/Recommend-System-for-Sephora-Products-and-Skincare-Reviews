import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
from wordcloud import WordCloud
from PIL import Image

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Download NLTK resources if needed
# nltk.download('punkt')
# nltk.download('wordnet')

class DirectTextAnalyzer:
    """
    A text analysis system that extracts words directly from reviews without standard stopwords,
    categorizes them into sentiment bags, and analyzes common themes in positive and negative reviews.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the analyzer with either a path to CSV data or a pandas DataFrame.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing review data
        df : pandas.DataFrame, optional
            DataFrame containing review data
        """
        # Initialize lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom minimal stopwords - only remove most basic words
        self.minimal_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'no', 'yes', 'this', 'that', 'these', 'those'}
        
        # Load data
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df
        else:
            self.df = None
            print("No data provided. Use load_data() to load data later.")
        
        # Initialize sentiment word bags
        self.positive_words = []
        self.negative_words = []
        self.neutral_words = []
        
        # Common skincare problem indicators
        self.problem_indicators = {
            'dryness': ['dry', 'drying', 'dried', 'dehydrated', 'flaky', 'chapped', 'parched'],
            'irritation': ['irritate', 'irritation', 'irritated', 'burn', 'burning', 'sting', 'stinging', 'itch', 'itchy', 'itching', 'red', 'redness'],
            'breakouts': ['breakout', 'pimple', 'acne', 'zit', 'blemish', 'clog', 'clogged', 'blackhead', 'whitehead', 'bump'],
            'sensitivity': ['sensitive', 'sensitivity', 'react', 'reaction', 'allergy', 'allergic'],
            'texture_issues': ['sticky', 'greasy', 'oily', 'thick', 'thin', 'watery', 'consistency', 'lumpy', 'gritty', 'grainy'],
            'absorption': ['absorb', 'absorption', 'sit', 'surface', 'soak', 'heavy'],
            'smell_issues': ['smell', 'scent', 'fragrance', 'odor', 'perfume', 'strong'],
            'efficacy': ['ineffective', 'useless', 'waste', 'didn\'t work', 'doesn\'t work', 'no difference', 'no change', 'no improvement'],
            'packaging': ['packaging', 'container', 'bottle', 'jar', 'pump', 'dispenser', 'applicator', 'broke', 'broken', 'leaking', 'messy'],
            'value': ['expensive', 'pricey', 'overpriced', 'cost', 'waste', 'money', 'worth', 'price', 'cheap']
        }
    
    def load_data(self, data_path=None, df=None):
        """
        Load review data from a CSV file or DataFrame.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file
        df : pandas.DataFrame, optional
            DataFrame containing review data
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Please provide either a data path or DataFrame")
        
        print(f"Loaded data with {len(self.df)} reviews.")
        print(f"Columns: {', '.join(self.df.columns)}")
    
    def clean_and_tokenize(self, text):
        """
        Clean and tokenize text without removing most words.
        
        Parameters:
        -----------
        text : str
            Text to clean and tokenize
            
        Returns:
        --------
        list
            List of tokenized words
        """
        if not isinstance(text, str):
            return []
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = text.strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove only minimal stopwords and very short words
        tokens = [t for t in tokens if t.isalpha() and t not in self.minimal_stopwords and len(t) > 1]
        
        # Lemmatize to normalize words
        lemmatized = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return lemmatized
    
    def extract_word_bags(self, text_column='review_text', rating_column='rating', 
                         sentiment_threshold=3, include_neutral=True):
        """
        Extract words into positive, negative, and neutral bags based on ratings.
        
        Parameters:
        -----------
        text_column : str, default='review_text'
            Column containing review text
        rating_column : str, default='rating'
            Column containing ratings
        sentiment_threshold : int, default=3
            Rating threshold for positive/negative sentiment
        include_neutral : bool, default=True
            Whether to include neutral words
            
        Returns:
        --------
        dict
            Dictionary containing word bags and frequencies
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
            
        if text_column not in self.df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data.")
            
        if rating_column not in self.df.columns:
            raise ValueError(f"Rating column '{rating_column}' not found in data.")
        
        print("Extracting word bags from reviews...")
        
        # Initialize word counters
        positive_counter = Counter()
        negative_counter = Counter()
        neutral_counter = Counter()
        
        # Process each review
        for _, row in self.df.iterrows():
            text = row[text_column]
            rating = row[rating_column]
            
            # Skip reviews with missing text or rating
            if pd.isna(text) or pd.isna(rating):
                continue
                
            # Tokenize the text
            tokens = self.clean_and_tokenize(text)
            
            # Categorize words based on rating
            if rating > sentiment_threshold:
                positive_counter.update(tokens)
            elif rating < sentiment_threshold:
                negative_counter.update(tokens)
            elif include_neutral:
                neutral_counter.update(tokens)
        
        # Store word bags
        self.positive_words = positive_counter
        self.negative_words = negative_counter
        self.neutral_words = neutral_counter
        
        # Create word bag summary
        word_bags = {
            'positive': positive_counter,
            'negative': negative_counter,
            'neutral': neutral_counter
        }
        
        print(f"Extracted {len(positive_counter)} unique positive words, "
              f"{len(negative_counter)} unique negative words, and "
              f"{len(neutral_counter)} unique neutral words.")
        
        return word_bags
    
    def analyze_word_sentiment_distribution(self, min_frequency=2, top_n=20):
        """
        Analyze the distribution of words across sentiment categories.
        
        Parameters:
        -----------
        min_frequency : int, default=2
            Minimum frequency for including words
        top_n : int, default=20
            Number of top words to show
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        if not self.positive_words or not self.negative_words:
            raise ValueError("Word bags not extracted. Run extract_word_bags() first.")
            
        print("Analyzing word sentiment distribution...")
        
        # Get top words by sentiment
        top_positive = [(word, count) for word, count in self.positive_words.most_common()
                        if count >= min_frequency][:top_n]
        
        top_negative = [(word, count) for word, count in self.negative_words.most_common()
                        if count >= min_frequency][:top_n]
        
        # Find words that appear in both positive and negative reviews
        all_positive_words = set(word for word, count in self.positive_words.items() 
                               if count >= min_frequency)
        all_negative_words = set(word for word, count in self.negative_words.items() 
                               if count >= min_frequency)
        
        common_words = all_positive_words.intersection(all_negative_words)
        
        # Calculate sentiment bias for common words
        polarized_words = []
        for word in common_words:
            pos_count = self.positive_words[word]
            neg_count = self.negative_words[word]
            total = pos_count + neg_count
            
            if total >= min_frequency:
                pos_ratio = pos_count / total
                sentiment_bias = (pos_count - neg_count) / total  # -1 to 1 scale
                polarized_words.append((word, pos_count, neg_count, sentiment_bias))
        
        # Sort by absolute sentiment bias
        polarized_words.sort(key=lambda x: abs(x[3]), reverse=True)
        
        # Prepare results
        results = {
            'top_positive_words': top_positive,
            'top_negative_words': top_negative,
            'polarized_words': polarized_words[:top_n]
        }
        
        # Print some insights
        print("\nTop positive words:")
        for word, count in top_positive[:10]:
            print(f"  {word}: {count}")
            
        print("\nTop negative words:")
        for word, count in top_negative[:10]:
            print(f"  {word}: {count}")
            
        print("\nWords with strong sentiment bias (appearing in both positive and negative reviews):")
        for word, pos_count, neg_count, bias in polarized_words[:10]:
            bias_direction = "positive" if bias > 0 else "negative"
            print(f"  {word}: {bias:.2f} bias towards {bias_direction} (pos: {pos_count}, neg: {neg_count})")
        
        return results
    
    def extract_ngrams(self, n=2, min_frequency=2, top_n=20, by_sentiment=True):
        """
        Extract common n-grams from reviews, optionally by sentiment.
        
        Parameters:
        -----------
        n : int, default=2
            N-gram size
        min_frequency : int, default=2
            Minimum frequency for including n-grams
        top_n : int, default=20
            Number of top n-grams to show
        by_sentiment : bool, default=True
            Whether to split n-grams by sentiment
            
        Returns:
        --------
        dict
            Dictionary containing n-gram analysis results
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
            
        print(f"Extracting {n}-grams from reviews...")
        
        # Initialize n-gram counters
        positive_ngrams = Counter()
        negative_ngrams = Counter()
        all_ngrams = Counter()
        
        # Process each review
        for _, row in self.df.iterrows():
            if 'review_text' not in row or pd.isna(row['review_text']):
                continue
                
            # Tokenize the text
            tokens = self.clean_and_tokenize(row['review_text'])
            
            # Skip if too few tokens
            if len(tokens) < n:
                continue
                
            # Generate n-grams
            review_ngrams = list(ngrams(tokens, n))
            
            # Convert n-gram tuples to strings
            review_ngrams = [' '.join(gram) for gram in review_ngrams]
            
            # Update counters
            all_ngrams.update(review_ngrams)
            
            # Update sentiment-specific counters if applicable
            if by_sentiment and 'rating' in row and not pd.isna(row['rating']):
                if row['rating'] > 3:
                    positive_ngrams.update(review_ngrams)
                elif row['rating'] < 3:
                    negative_ngrams.update(review_ngrams)
        
        # Filter by minimum frequency
        all_ngrams = [(gram, count) for gram, count in all_ngrams.most_common()
                     if count >= min_frequency][:top_n]
        
        positive_ngrams = [(gram, count) for gram, count in positive_ngrams.most_common()
                          if count >= min_frequency][:top_n]
        
        negative_ngrams = [(gram, count) for gram, count in negative_ngrams.most_common()
                          if count >= min_frequency][:top_n]
        
        # Prepare results
        results = {
            'all_ngrams': all_ngrams,
            'positive_ngrams': positive_ngrams,
            'negative_ngrams': negative_ngrams
        }
        
        # Print some insights
        print(f"\nTop {n}-grams across all reviews:")
        for gram, count in all_ngrams[:10]:
            print(f"  '{gram}': {count}")
            
        if by_sentiment:
            print(f"\nTop {n}-grams in positive reviews:")
            for gram, count in positive_ngrams[:10]:
                print(f"  '{gram}': {count}")
                
            print(f"\nTop {n}-grams in negative reviews:")
            for gram, count in negative_ngrams[:10]:
                print(f"  '{gram}': {count}")
        
        return results
    
    def analyze_common_problems(self, text_column='review_text', min_frequency=1):
        """
        Analyze common skincare problems mentioned in reviews.
        
        Parameters:
        -----------
        text_column : str, default='review_text'
            Column containing review text
        min_frequency : int, default=1
            Minimum frequency for including problems
            
        Returns:
        --------
        dict
            Dictionary containing problem analysis results
        """
        if self.df is None or text_column not in self.df.columns:
            raise ValueError("Data not properly loaded or missing text column.")
            
        print("Analyzing common skincare problems...")
        
        # Initialize problem counters
        problem_counts = {category: 0 for category in self.problem_indicators}
        problem_reviews = {category: [] for category in self.problem_indicators}
        
        # Process each review
        for idx, row in self.df.iterrows():
            if pd.isna(row[text_column]):
                continue
                
            text = row[text_column].lower()
            rating = row['rating'] if 'rating' in row and not pd.isna(row['rating']) else None
            
            # Check for each problem category
            for category, indicators in self.problem_indicators.items():
                for indicator in indicators:
                    if indicator in text:
                        problem_counts[category] += 1
                        
                        # Store review details
                        problem_reviews[category].append({
                            'review_id': idx,
                            'text': text,
                            'rating': rating,
                            'indicator': indicator
                        })
                        
                        # Only count each category once per review
                        break
        
        # Filter by minimum frequency
        filtered_problems = {category: count for category, count in problem_counts.items()
                           if count >= min_frequency}
        
        # Sort problems by frequency
        sorted_problems = sorted(filtered_problems.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = {
            'problem_counts': sorted_problems,
            'problem_reviews': problem_reviews
        }
        
        # Print insights
        print("\nCommon skincare problems mentioned:")
        for category, count in sorted_problems:
            print(f"  {category}: {count} mentions")
            
            # Show example reviews (up to 2)
            if problem_reviews[category]:
                for i, review in enumerate(problem_reviews[category][:2]):
                    print(f"    Example {i+1}: \"{review['text'][:100]}...\"")
        
        return results
    
    def analyze_product_specific_issues(self, text_column='review_text', 
                                       product_column='product_name', 
                                       min_reviews=2):
        """
        Analyze product-specific issues and patterns.
        
        Parameters:
        -----------
        text_column : str, default='review_text'
            Column containing review text
        product_column : str, default='product_name'
            Column containing product names
        min_reviews : int, default=2
            Minimum reviews for a product to be analyzed
            
        Returns:
        --------
        dict
            Dictionary containing product-specific analysis results
        """
        if (self.df is None or text_column not in self.df.columns or
            product_column not in self.df.columns):
            raise ValueError("Data not properly loaded or missing required columns.")
            
        print("Analyzing product-specific issues...")
        
        # Get product counts
        product_counts = self.df[product_column].value_counts()
        valid_products = product_counts[product_counts >= min_reviews].index
        
        # Initialize results
        product_analysis = {}
        
        # Analyze each product
        for product in valid_products:
            # Get reviews for this product
            product_df = self.df[self.df[product_column] == product]
            
            # Skip products with too few reviews
            if len(product_df) < min_reviews:
                continue
                
            # Calculate average rating if available
            avg_rating = None
            if 'rating' in product_df.columns:
                avg_rating = product_df['rating'].mean()
                
            # Extract words from positive and negative reviews
            pos_words = Counter()
            neg_words = Counter()
            
            for _, row in product_df.iterrows():
                if pd.isna(row[text_column]):
                    continue
                    
                tokens = self.clean_and_tokenize(row[text_column])
                
                if 'rating' in row and not pd.isna(row['rating']):
                    if row['rating'] > 3:
                        pos_words.update(tokens)
                    elif row['rating'] < 3:
                        neg_words.update(tokens)
            
            # Analyze problems for this product
            product_problems = {category: 0 for category in self.problem_indicators}
            
            for _, row in product_df.iterrows():
                if pd.isna(row[text_column]):
                    continue
                    
                text = row[text_column].lower()
                
                for category, indicators in self.problem_indicators.items():
                    for indicator in indicators:
                        if indicator in text:
                            product_problems[category] += 1
                            break
            
            # Store product analysis
            product_analysis[product] = {
                'review_count': len(product_df),
                'avg_rating': avg_rating,
                'positive_words': pos_words.most_common(10),
                'negative_words': neg_words.most_common(10),
                'problems': sorted(product_problems.items(), key=lambda x: x[1], reverse=True)
            }
        
        # Print insights
        print("\nProduct-specific analysis:")
        for product, analysis in product_analysis.items():
            print(f"\n  {product} ({analysis['review_count']} reviews):")
            
            if analysis['avg_rating'] is not None:
                print(f"    Average rating: {analysis['avg_rating']:.1f}")
                
            print("    Top positive words:", ", ".join([word for word, _ in analysis['positive_words'][:5]]))
            print("    Top negative words:", ", ".join([word for word, _ in analysis['negative_words'][:5]]))
            print("    Main problems:", ", ".join([f"{category} ({count})" for category, count in analysis['problems'][:3] if count > 0]))
        
        return product_analysis
    
    def visualize_word_cloud(self, sentiment='positive', min_frequency=1, filename=None):
        """
        Generate word cloud for specific sentiment.
        
        Parameters:
        -----------
        sentiment : str, default='positive'
            Sentiment category ('positive', 'negative', or 'neutral')
        min_frequency : int, default=1
            Minimum word frequency to include
        filename : str, optional
            If provided, save the word cloud to this file
            
        Returns:
        --------
        matplotlib.figure.Figure
            The word cloud figure
        """
        if not hasattr(self, 'positive_words') or not self.positive_words:
            raise ValueError("Word bags not extracted. Run extract_word_bags() first.")
            
        # Select word bag based on sentiment
        if sentiment == 'positive':
            words = self.positive_words
            colormap = 'YlGn'
            title = 'Positive Review Words'
        elif sentiment == 'negative':
            words = self.negative_words
            colormap = 'OrRd'
            title = 'Negative Review Words'
        elif sentiment == 'neutral':
            words = self.neutral_words
            colormap = 'Blues'
            title = 'Neutral Review Words'
        else:
            raise ValueError("Invalid sentiment. Choose 'positive', 'negative', or 'neutral'.")
            
        # Filter by minimum frequency
        filtered_words = {word: count for word, count in words.items() if count >= min_frequency}
        
        # Check if there are words to visualize
        if not filtered_words:
            print(f"No {sentiment} words with frequency >= {min_frequency}.")
            return None
            
        # Create word cloud
        wc = WordCloud(
            background_color='white',
            max_words=100,
            colormap=colormap,
            width=800,
            height=400,
            contour_width=1,
            contour_color='steelblue'
        ).generate_from_frequencies(filtered_words)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Word cloud saved to {filename}")
            
        return plt.gcf()
    
    def visualize_common_problems(self, filename=None):
        """
        Visualize common skincare problems mentioned in reviews.
        
        Parameters:
        -----------
        filename : str, optional
            If provided, save the visualization to this file
            
        Returns:
        --------
        matplotlib.figure.Figure
            The visualization figure
        """
        if not hasattr(self, 'analyze_common_problems'):
            problems = self.analyze_common_problems()
        else:
            problems = self.analyze_common_problems()
            
        # Extract problem counts
        problem_counts = problems['problem_counts']
        
        if not problem_counts:
            print("No problem counts to visualize.")
            return None
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Extract categories and counts
        categories, counts = zip(*problem_counts)
        
        # Create horizontal bar chart
        plt.barh(categories, counts, color='salmon')
        plt.xlabel('Number of Mentions')
        plt.ylabel('Problem Category')
        plt.title('Common Skincare Problems Mentioned in Reviews', fontsize=16)
        
        # Add counts as labels
        for i, count in enumerate(counts):
            plt.text(count + 0.1, i, str(count), va='center')
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Problem visualization saved to {filename}")
            
        return plt.gcf()
    
    def visualize_sentiment_comparison(self, top_n=15, filename=None):
        """
        Create a comparison of top words in positive and negative reviews.
        
        Parameters:
        -----------
        top_n : int, default=15
            Number of top words to compare
        filename : str, optional
            If provided, save the visualization to this file
            
        Returns:
        --------
        matplotlib.figure.Figure
            The visualization figure
        """
        if not hasattr(self, 'positive_words') or not self.positive_words:
            raise ValueError("Word bags not extracted. Run extract_word_bags() first.")
            
        # Get top words
        top_positive = self.positive_words.most_common(top_n)
        top_negative = self.negative_words.most_common(top_n)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Positive words
        if top_positive:
            words, counts = zip(*reversed(top_positive))
            ax1.barh(words, counts, color='forestgreen')
            ax1.set_title('Top Words in Positive Reviews', fontsize=14)
            ax1.set_xlabel('Frequency')
            
        # Negative words
        if top_negative:
            words, counts = zip(*reversed(top_negative))
            ax2.barh(words, counts, color='firebrick')
            ax2.set_title('Top Words in Negative Reviews', fontsize=14)
            ax2.set_xlabel('Frequency')
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Sentiment comparison saved to {filename}")
            
        return fig
    
    def generate_comprehensive_analysis(self, output_dir='.', format='html'):
        """
        Generate a comprehensive text analysis report.
        
        Parameters:
        -----------
        output_dir : str, default='.'
            Directory to save the report
        format : str, default='html'
            Report format ('html', 'markdown', or 'text')
            
        Returns:
        --------
        str
            Path to the generated report
        """
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"skincare_text_analysis_{timestamp}.{format}"
        filepath = os.path.join(output_dir, filename)
        
        # Initialize report content
        if format == 'html':
            report = ["<html><head>",
                     "<title>Comprehensive Skincare Text Analysis</title>",
                     "<style>",
                     "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }",
                     "h1, h2, h3 { color: #333366; }",
                     "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
                     "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }",
                     "th { background-color: #f2f2f2; }",
                     "tr:nth-child(even) { background-color: #f9f9f9; }",
                     ".positive { color: green; }",
                     ".negative { color: red; }",
                     ".neutral { color: gray; }",
                     ".column { float: left; width: 48%; margin-right: 2%; }",
                     ".row:after { content: \"\"; display: table; clear: both; }",
                     "</style>",
                     "</head><body>",
                     f"<h1>Comprehensive Skincare Text Analysis</h1>",
                     f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"]
        elif format == 'markdown':
            report = [f"# Comprehensive Skincare Text Analysis",
                     f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        else:  # text
            report = [f"COMPREHENSIVE SKINCARE TEXT ANALYSIS",
                     f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     "=" * 50]
            
        # Ensure we have the necessary analyses
        if not hasattr(self, 'positive_words') or not self.positive_words:
            self.extract_word_bags()
            
        word_sentiment = self.analyze_word_sentiment_distribution()
        bigrams = self.extract_ngrams(n=2)
        problems = self.analyze_common_problems()
        
        if 'product_name' in self.df.columns:
            product_analysis = self.analyze_product_specific_issues()
        else:
            product_analysis = None
            
        # Add dataset overview
        if format == 'html':
            report.append("<h2>1. Dataset Overview</h2>")
            report.append(f"<p>Total reviews analyzed: {len(self.df)}</p>")
            
            if 'product_name' in self.df.columns:
                products = self.df['product_name'].value_counts()
                report.append("<h3>Products in the Dataset</h3><ul>")
                for product, count in products.items():
                    report.append(f"<li>{product}: {count} reviews</li>")
                report.append("</ul>")
                
            if 'rating' in self.df.columns:
                report.append("<h3>Rating Distribution</h3>")
                rating_counts = self.df['rating'].value_counts().sort_index()
                report.append("<table>")
                report.append("<tr><th>Rating</th><th>Count</th><th>Percentage</th></tr>")
                
                for rating, count in rating_counts.items():
                    pct = (count / len(self.df)) * 100
                    report.append(f"<tr><td>{rating}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
                    
                report.append("</table>")
        elif format == 'markdown':
            report.append("## 1. Dataset Overview")
            report.append(f"Total reviews analyzed: {len(self.df)}")
            
            if 'product_name' in self.df.columns:
                products = self.df['product_name'].value_counts()
                report.append("\n### Products in the Dataset")
                for product, count in products.items():
                    report.append(f"- {product}: {count} reviews")
                
            if 'rating' in self.df.columns:
                report.append("\n### Rating Distribution")
                report.append("\n| Rating | Count | Percentage |")
                report.append("| --- | --- | --- |")
                
                rating_counts = self.df['rating'].value_counts().sort_index()
                for rating, count in rating_counts.items():
                    pct = (count / len(self.df)) * 100
                    report.append(f"| {rating} | {count} | {pct:.1f}% |")
        else:  # text
            report.append("\n1. DATASET OVERVIEW")
            report.append("-" * 30)
            report.append(f"Total reviews analyzed: {len(self.df)}")
            
            if 'product_name' in self.df.columns:
                products = self.df['product_name'].value_counts()
                report.append("\nProducts in the Dataset:")
                for product, count in products.items():
                    report.append(f"- {product}: {count} reviews")
                
            if 'rating' in self.df.columns:
                report.append("\nRating Distribution:")
                rating_counts = self.df['rating'].value_counts().sort_index()
                for rating, count in rating_counts.items():
                    pct = (count / len(self.df)) * 100
                    report.append(f"Rating {rating}: {count} ({pct:.1f}%)")
                    
        # Add word sentiment analysis
        if format == 'html':
            report.append("<h2>2. What Customers Say Most Often</h2>")
            
            # Positive reviews section
            report.append("<h3>2.1 Most Common Words in Positive Reviews</h3>")
            report.append("<div class='row'>")
            report.append("<div class='column'>")
            report.append("<table>")
            report.append("<tr><th>Word</th><th>Frequency</th></tr>")
            
            for word, count in word_sentiment['top_positive_words'][:15]:
                report.append(f"<tr><td>{word}</td><td>{count}</td></tr>")
                
            report.append("</table>")
            report.append("</div>")
            
            # Add positive bigrams in second column
            report.append("<div class='column'>")
            report.append("<h4>Common Phrases in Positive Reviews</h4>")
            report.append("<table>")
            report.append("<tr><th>Phrase</th><th>Frequency</th></tr>")
            
            for phrase, count in bigrams['positive_ngrams'][:10]:
                report.append(f"<tr><td>{phrase}</td><td>{count}</td></tr>")
                
            report.append("</table>")
            report.append("</div>")
            report.append("</div>")
            
            # Negative reviews section
            report.append("<h3>2.2 Most Common Words in Negative Reviews</h3>")
            report.append("<div class='row'>")
            report.append("<div class='column'>")
            report.append("<table>")
            report.append("<tr><th>Word</th><th>Frequency</th></tr>")
            
            for word, count in word_sentiment['top_negative_words'][:15]:
                report.append(f"<tr><td>{word}</td><td>{count}</td></tr>")
                
            report.append("</table>")
            report.append("</div>")
            
            # Add negative bigrams in second column
            report.append("<div class='column'>")
            report.append("<h4>Common Phrases in Negative Reviews</h4>")
            report.append("<table>")
            report.append("<tr><th>Phrase</th><th>Frequency</th></tr>")
            
            for phrase, count in bigrams['negative_ngrams'][:10]:
                report.append(f"<tr><td>{phrase}</td><td>{count}</td></tr>")
                
            report.append("</table>")
            report.append("</div>")
            report.append("</div>")
            
            # Polarized words section
            report.append("<h3>2.3 Words with Strong Sentiment Bias</h3>")
            report.append("<p>These words appear in both positive and negative reviews but show a strong bias toward one sentiment.</p>")
            report.append("<table>")
            report.append("<tr><th>Word</th><th>Positive Uses</th><th>Negative Uses</th><th>Sentiment Bias</th></tr>")
            
            for word, pos_count, neg_count, bias in word_sentiment['polarized_words'][:15]:
                bias_class = "positive" if bias > 0 else "negative"
                report.append(f"<tr><td>{word}</td><td>{pos_count}</td><td>{neg_count}</td>" +
                             f"<td class='{bias_class}'>{bias:.2f}</td></tr>")
                
            report.append("</table>")
        elif format == 'markdown':
            report.append("\n## 2. What Customers Say Most Often")
            
            # Positive reviews section
            report.append("\n### 2.1 Most Common Words in Positive Reviews")
            report.append("\n| Word | Frequency |")
            report.append("| --- | --- |")
            
            for word, count in word_sentiment['top_positive_words'][:15]:
                report.append(f"| {word} | {count} |")
                
            report.append("\n#### Common Phrases in Positive Reviews")
            report.append("\n| Phrase | Frequency |")
            report.append("| --- | --- |")
            
            for phrase, count in bigrams['positive_ngrams'][:10]:
                report.append(f"| {phrase} | {count} |")
                
            # Negative reviews section
            report.append("\n### 2.2 Most Common Words in Negative Reviews")
            report.append("\n| Word | Frequency |")
            report.append("| --- | --- |")
            
            for word, count in word_sentiment['top_negative_words'][:15]:
                report.append(f"| {word} | {count} |")
                
            report.append("\n#### Common Phrases in Negative Reviews")
            report.append("\n| Phrase | Frequency |")
            report.append("| --- | --- |")
            
            for phrase, count in bigrams['negative_ngrams'][:10]:
                report.append(f"| {phrase} | {count} |")
                
            # Polarized words section
            report.append("\n### 2.3 Words with Strong Sentiment Bias")
            report.append("\nThese words appear in both positive and negative reviews but show a strong bias toward one sentiment.")
            report.append("\n| Word | Positive Uses | Negative Uses | Sentiment Bias |")
            report.append("| --- | --- | --- | --- |")
            
            for word, pos_count, neg_count, bias in word_sentiment['polarized_words'][:15]:
                report.append(f"| {word} | {pos_count} | {neg_count} | {bias:.2f} |")
        else:  # text
            report.append("\n2. WHAT CUSTOMERS SAY MOST OFTEN")
            report.append("-" * 30)
            
            # Positive reviews section
            report.append("\n2.1 Most Common Words in Positive Reviews:")
            for word, count in word_sentiment['top_positive_words'][:15]:
                report.append(f"- {word}: {count}")
                
            report.append("\nCommon Phrases in Positive Reviews:")
            for phrase, count in bigrams['positive_ngrams'][:10]:
                report.append(f"- '{phrase}': {count}")
                
            # Negative reviews section
            report.append("\n2.2 Most Common Words in Negative Reviews:")
            for word, count in word_sentiment['top_negative_words'][:15]:
                report.append(f"- {word}: {count}")
                
            report.append("\nCommon Phrases in Negative Reviews:")
            for phrase, count in bigrams['negative_ngrams'][:10]:
                report.append(f"- '{phrase}': {count}")
                
            # Polarized words section
            report.append("\n2.3 Words with Strong Sentiment Bias:")
            report.append("These words appear in both positive and negative reviews but show a strong bias toward one sentiment.")
            
            for word, pos_count, neg_count, bias in word_sentiment['polarized_words'][:15]:
                bias_direction = "positive" if bias > 0 else "negative"
                report.append(f"- {word}: {bias:.2f} bias towards {bias_direction} (pos: {pos_count}, neg: {neg_count})")
                
        # Add common skincare problems analysis
        if format == 'html':
            report.append("<h2>3. Common Skincare Problems</h2>")
            report.append("<p>Analysis of skincare issues and concerns mentioned in reviews.</p>")
            
            # Problem frequency table
            report.append("<h3>3.1 Problem Frequency</h3>")
            report.append("<table>")
            report.append("<tr><th>Problem Category</th><th>Mentions</th><th>Examples</th></tr>")
            
            for category, count in problems['problem_counts']:
                # Get example reviews
                examples = problems['problem_reviews'][category][:2]
                example_text = ""
                
                for ex in examples:
                    truncated_text = ex['text'][:100] + "..." if len(ex['text']) > 100 else ex['text']
                    example_text += f"<p><em>\"{truncated_text}\"</em></p>"
                    
                report.append(f"<tr><td>{category}</td><td>{count}</td><td>{example_text}</td></tr>")
                
            report.append("</table>")
            
            # Product-specific problems if available
            if product_analysis:
                report.append("<h3>3.2 Product-Specific Issues</h3>")
                
                for product, analysis in product_analysis.items():
                    report.append(f"<h4>{product}</h4>")
                    report.append("<table>")
                    report.append("<tr><th>Problem Category</th><th>Mentions</th></tr>")
                    
                    for category, count in analysis['problems']:
                        if count > 0:
                            report.append(f"<tr><td>{category}</td><td>{count}</td></tr>")
                            
                    report.append("</table>")
        elif format == 'markdown':
            report.append("\n## 3. Common Skincare Problems")
            report.append("\nAnalysis of skincare issues and concerns mentioned in reviews.")
            
            # Problem frequency table
            report.append("\n### 3.1 Problem Frequency")
            report.append("\n| Problem Category | Mentions | Example |")
            report.append("| --- | --- | --- |")
            
            for category, count in problems['problem_counts']:
                # Get first example review
                examples = problems['problem_reviews'][category]
                if examples:
                    truncated_text = examples[0]['text'][:80] + "..." if len(examples[0]['text']) > 80 else examples[0]['text']
                    report.append(f"| {category} | {count} | \"{truncated_text}\" |")
                else:
                    report.append(f"| {category} | {count} | - |")
                    
            # Product-specific problems if available
            if product_analysis:
                report.append("\n### 3.2 Product-Specific Issues")
                
                for product, analysis in product_analysis.items():
                    report.append(f"\n#### {product}")
                    report.append("\n| Problem Category | Mentions |")
                    report.append("| --- | --- |")
                    
                    for category, count in analysis['problems']:
                        if count > 0:
                            report.append(f"| {category} | {count} |")
        else:  # text
            report.append("\n3. COMMON SKINCARE PROBLEMS")
            report.append("-" * 30)
            report.append("\nAnalysis of skincare issues and concerns mentioned in reviews.")
            
            # Problem frequency
            report.append("\n3.1 Problem Frequency:")
            for category, count in problems['problem_counts']:
                report.append(f"- {category}: {count} mentions")
                
                # Add example
                examples = problems['problem_reviews'][category]
                if examples:
                    truncated_text = examples[0]['text'][:80] + "..." if len(examples[0]['text']) > 80 else examples[0]['text']
                    report.append(f"  Example: \"{truncated_text}\"")
                    
            # Product-specific problems if available
            if product_analysis:
                report.append("\n3.2 Product-Specific Issues:")
                
                for product, analysis in product_analysis.items():
                    report.append(f"\n{product}:")
                    for category, count in analysis['problems']:
                        if count > 0:
                            report.append(f"- {category}: {count} mentions")
                            
        # Add key insights and conclusion
        if format == 'html':
            report.append("<h2>4. Key Insights and Recommendations</h2>")
            
            # Generate insights based on the analysis
            pos_words = [word for word, _ in word_sentiment['top_positive_words'][:10]]
            neg_words = [word for word, _ in word_sentiment['top_negative_words'][:10]]
            top_problems = [category for category, _ in problems['problem_counts'][:5]]
            
            report.append("<h3>4.1 Positive Aspects Customers Value</h3>")
            report.append("<ul>")
            report.append(f"<li>Customers consistently praise products for being <strong>{', '.join(pos_words[:3])}</strong>.</li>")
            
            # Add insight about phrases
            if bigrams['positive_ngrams']:
                pos_phrases = [phrase for phrase, _ in bigrams['positive_ngrams'][:3]]
                report.append(f"<li>Common positive phrases like <strong>'{', '.join(pos_phrases)}'</strong> indicate what customers appreciate.</li>")
                
            report.append("</ul>")
            
            report.append("<h3>4.2 Areas for Improvement</h3>")
            report.append("<ul>")
            report.append(f"<li>Negative reviews frequently mention <strong>{', '.join(neg_words[:3])}</strong>.</li>")
            report.append(f"<li>The most common skincare problems reported are <strong>{', '.join(top_problems[:3])}</strong>.</li>")
            
            # Add insight about phrases
            if bigrams['negative_ngrams']:
                neg_phrases = [phrase for phrase, _ in bigrams['negative_ngrams'][:3]]
                report.append(f"<li>Phrases like <strong>'{', '.join(neg_phrases)}'</strong> indicate customer pain points.</li>")
                
            report.append("</ul>")
            
            report.append("<h3>4.3 Product-Specific Recommendations</h3>")
            
            if product_analysis:
                report.append("<ul>")
                
                for product, analysis in product_analysis.items():
                    top_problems = [category for category, count in analysis['problems'][:2] if count > 0]
                    
                    if top_problems:
                        report.append(f"<li><strong>{product}:</strong> Address {', '.join(top_problems)} issues based on customer feedback.</li>")
                        
                report.append("</ul>")
        elif format == 'markdown':
            report.append("\n## 4. Key Insights and Recommendations")
            
            # Generate insights based on the analysis
            pos_words = [word for word, _ in word_sentiment['top_positive_words'][:10]]
            neg_words = [word for word, _ in word_sentiment['top_negative_words'][:10]]
            top_problems = [category for category, _ in problems['problem_counts'][:5]]
            
            report.append("\n### 4.1 Positive Aspects Customers Value")
            report.append(f"\n- Customers consistently praise products for being **{', '.join(pos_words[:3])}**.")
            
            # Add insight about phrases
            if bigrams['positive_ngrams']:
                pos_phrases = [phrase for phrase, _ in bigrams['positive_ngrams'][:3]]
                report.append(f"- Common positive phrases like **'{', '.join(pos_phrases)}'** indicate what customers appreciate.")
                
            report.append("\n### 4.2 Areas for Improvement")
            report.append(f"\n- Negative reviews frequently mention **{', '.join(neg_words[:3])}**.")
            report.append(f"- The most common skincare problems reported are **{', '.join(top_problems[:3])}**.")
            
            # Add insight about phrases
            if bigrams['negative_ngrams']:
                neg_phrases = [phrase for phrase, _ in bigrams['negative_ngrams'][:3]]
                report.append(f"- Phrases like **'{', '.join(neg_phrases)}'** indicate customer pain points.")
                
            report.append("\n### 4.3 Product-Specific Recommendations")
            
            if product_analysis:
                for product, analysis in product_analysis.items():
                    top_problems = [category for category, count in analysis['problems'][:2] if count > 0]
                    
                    if top_problems:
                        report.append(f"\n- **{product}:** Address {', '.join(top_problems)} issues based on customer feedback.")
        else:  # text
            report.append("\n4. KEY INSIGHTS AND RECOMMENDATIONS")
            report.append("-" * 30)
            
            # Generate insights based on the analysis
            pos_words = [word for word, _ in word_sentiment['top_positive_words'][:10]]
            neg_words = [word for word, _ in word_sentiment['top_negative_words'][:10]]
            top_problems = [category for category, _ in problems['problem_counts'][:5]]
            
            report.append("\n4.1 Positive Aspects Customers Value:")
            report.append(f"- Customers consistently praise products for being {', '.join(pos_words[:3])}.")
            
            # Add insight about phrases
            if bigrams['positive_ngrams']:
                pos_phrases = [phrase for phrase, _ in bigrams['positive_ngrams'][:3]]
                report.append(f"- Common positive phrases like '{', '.join(pos_phrases)}' indicate what customers appreciate.")
                
            report.append("\n4.2 Areas for Improvement:")
            report.append(f"- Negative reviews frequently mention {', '.join(neg_words[:3])}.")
            report.append(f"- The most common skincare problems reported are {', '.join(top_problems[:3])}.")
            
            # Add insight about phrases
            if bigrams['negative_ngrams']:
                neg_phrases = [phrase for phrase, _ in bigrams['negative_ngrams'][:3]]
                report.append(f"- Phrases like '{', '.join(neg_phrases)}' indicate customer pain points.")
                
            report.append("\n4.3 Product-Specific Recommendations:")
            
            if product_analysis:
                for product, analysis in product_analysis.items():
                    top_problems = [category for category, count in analysis['problems'][:2] if count > 0]
                    
                    if top_problems:
                        report.append(f"- {product}: Address {', '.join(top_problems)} issues based on customer feedback.")
        
        # Finish report
        if format == 'html':
            report.append("</body></html>")
            
        # Write report to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"Comprehensive analysis report generated: {filepath}")
        return filepath


def analyze_skincare_reviews(file_path, output_dir='.'):
    """
    Convenience function to run a complete analysis on skincare reviews.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV/TSV file with review data
    output_dir : str, default='.'
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    import os
    import pandas as pd
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        
    # Initialize analyzer
    analyzer = DirectTextAnalyzer(df=df)
    
    # Extract word bags by sentiment
    word_bags = analyzer.extract_word_bags()
    
    # Analyze word sentiment distribution
    word_sentiment = analyzer.analyze_word_sentiment_distribution()
    
    # Extract n-grams
    bigrams = analyzer.extract_ngrams(n=2)
    trigrams = analyzer.extract_ngrams(n=3)
    
    # Analyze common problems
    problems = analyzer.analyze_common_problems()
    
    # Analyze product-specific issues if applicable
    if 'product_name' in df.columns:
        product_analysis = analyzer.analyze_product_specific_issues()
    else:
        product_analysis = None
        
    # Generate visualizations
    analyzer.visualize_word_cloud('positive', filename=os.path.join(output_dir, 'positive_words.png'))
    analyzer.visualize_word_cloud('negative', filename=os.path.join(output_dir, 'negative_words.png'))
    analyzer.visualize_common_problems(filename=os.path.join(output_dir, 'skincare_problems.png'))
    analyzer.visualize_sentiment_comparison(filename=os.path.join(output_dir, 'sentiment_comparison.png'))
    
    # Generate comprehensive report
    report_path = analyzer.generate_comprehensive_analysis(output_dir=output_dir, format='html')
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Main report: {report_path}")
    
    # Return results dictionary
    return {
        'word_bags': word_bags,
        'word_sentiment': word_sentiment,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'problems': problems,
        'product_analysis': product_analysis,
        'report_path': report_path
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'skincare_analysis_results'
        
        analyze_skincare_reviews(file_path, output_dir)
    else:
        print("Usage: python direct_text_analyzer.py <file_path> [output_directory]")