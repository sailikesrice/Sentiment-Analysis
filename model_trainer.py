# model_trainer.py
from vader_model import VaderSentimentAnalyzer
import pickle
import os

def create_model():
    """Create and save the sentiment analyzer"""
    print("Creating VADER sentiment analyzer...")
    
    model = VaderSentimentAnalyzer()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Model saved to models/sentiment_model.pkl")
    
    # Test on examples
    test_reviews = [
        "This movie was amazing! Best film I've seen all year.",
        "Terrible waste of time. Worst movie ever.",
        "It was okay, nothing special.",
        "Absolutely incredible! A masterpiece of cinema!",
        "Boring and predictable. Would not recommend."
    ]
    
    predictions = model.predict(test_reviews)
    probabilities = model.predict_proba(test_reviews)
    
    print("\n" + "="*60)
    print("Test predictions:")
    print("="*60)
    for review, pred, prob in zip(test_reviews, predictions, probabilities):
        sentiment = "Positive ✓" if pred == 1 else "Negative ✗"
        confidence = prob[pred] * 100
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} ({confidence:.1f}% confidence)")

if __name__ == "__main__":
    create_model()