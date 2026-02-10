# vader_model.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VaderSentimentAnalyzer:
    """Sentiment analyzer using VADER"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def predict(self, reviews):
        """Predict sentiment for reviews"""
        predictions = []
        for review in reviews:
            scores = self.analyzer.polarity_scores(review)
            # compound score >= 0.05 is positive, <= -0.05 is negative, else neutral
            if scores['compound'] >= 0.05:
                sentiment = 1  # positive
            else:
                sentiment = 0  # negative
            predictions.append(sentiment)
        return predictions
    
    def predict_proba(self, reviews):
        """Get probability scores"""
        probabilities = []
        for review in reviews:
            scores = self.analyzer.polarity_scores(review)
            # Use positive and negative scores directly
            positive_prob = scores['pos']
            negative_prob = scores['neg']
            
            # Normalize to sum to 1
            total = positive_prob + negative_prob
            if total > 0:
                positive_prob = positive_prob / total
                negative_prob = negative_prob / total
            else:
                positive_prob = 0.5
                negative_prob = 0.5
            
            probabilities.append([negative_prob, positive_prob])
        return probabilities