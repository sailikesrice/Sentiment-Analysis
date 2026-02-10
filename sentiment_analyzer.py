# sentiment_analyzer.py
import pickle
import os
from vader_model import VaderSentimentAnalyzer

class SentimentAnalyzer:
    def __init__(self, model_path='models/sentiment_model.pkl'):
        """Load the pre-trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run model_trainer.py first."
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def analyze_single(self, review_text):
        """Analyze a single review"""
        prediction = self.model.predict([review_text])[0]
        probabilities = self.model.predict_proba([review_text])[0]
        
        return {
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(probabilities[prediction]),
            'positive_probability': float(probabilities[1]),
            'negative_probability': float(probabilities[0])
        }
    
    def analyze_batch(self, reviews):
        """Analyze multiple reviews and return aggregated results"""
        if not reviews:
            return {
                'error': 'No reviews provided',
                'total_reviews': 0
            }
        
        predictions = self.model.predict(reviews)
        probabilities = self.model.predict_proba(reviews)
        
        # Calculate statistics
        positive_count = int(sum(predictions))
        negative_count = len(predictions) - positive_count
        total = len(reviews)
        
        # Calculate average confidence
        confidences = []
        for pred, prob in zip(predictions, probabilities):
            confidence = prob[pred]
            confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Detailed results for each review
        detailed_results = []
        for i, (review, pred, prob) in enumerate(zip(reviews, predictions, probabilities)):
            detailed_results.append({
                'review_number': i + 1,
                'review_text': review[:200] + '...' if len(review) > 200 else review,
                'sentiment': 'positive' if pred == 1 else 'negative',
                'confidence': float(prob[pred]),
                'positive_probability': float(prob[1]),
                'negative_probability': float(prob[0])
            })
        
        return {
            'total_reviews': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_percentage': round((positive_count / total) * 100, 2) if total > 0 else 0,
            'negative_percentage': round((negative_count / total) * 100, 2) if total > 0 else 0,
            'average_confidence': round(float(avg_confidence), 3),
            'detailed_results': detailed_results
        }
    
    def find_positive_and_negative_examples(self, reviews_data):
        """Find the best positive and negative review examples"""
        # Handle empty reviews
        if not reviews_data or len(reviews_data) == 0:
            print("No reviews data provided")
            return None, None
        
        print(f"Processing {len(reviews_data)} reviews...")
        
        # Analyze all reviews
        review_texts = [r['content'] for r in reviews_data]
        predictions = self.model.predict(review_texts)
        probabilities = self.model.predict_proba(review_texts)
        
        # Find positive reviews (prediction = 1)
        positive_reviews = []
        negative_reviews = []
        
        for i, (review_data, pred, prob) in enumerate(zip(reviews_data, predictions, probabilities)):
            sentiment_info = {
                'author': review_data['author'],
                'content': review_data['content'],
                'sentiment': 'positive' if pred == 1 else 'negative',
                'confidence': float(prob[pred]),
                'positive_probability': float(prob[1]),
                'negative_probability': float(prob[0])
            }
            
            if pred == 1:  # Positive
                positive_reviews.append(sentiment_info)
            else:  # Negative
                negative_reviews.append(sentiment_info)
        
        print(f"Found {len(positive_reviews)} positive and {len(negative_reviews)} negative reviews")
        
        # Sort by confidence and get the most confident examples
        positive_reviews.sort(key=lambda x: x['confidence'], reverse=True)
        negative_reviews.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get best examples or None if not available
        best_positive = positive_reviews[0] if positive_reviews else None
        best_negative = negative_reviews[0] if negative_reviews else None
        
        # If we don't have both, create fallback message
        if best_positive is None:
            print("Warning: No positive reviews found")
        if best_negative is None:
            print("Warning: No negative reviews found")
        
        return best_positive, best_negative