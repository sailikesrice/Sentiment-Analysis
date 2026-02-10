# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_analyzer import SentimentAnalyzer
from tmdb_api import search_movie, get_movie_details, get_movie_reviews
import traceback

app = Flask(__name__)
CORS(app)

# Initialize sentiment analyzer
try:
    analyzer = SentimentAnalyzer()
    print("✓ Sentiment analyzer loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run model_trainer.py first to train the model")
    exit(1)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by title"""
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        results = search_movie(query)
        return jsonify({
            'results': results[:10],
            'total': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get movie details"""
    try:
        details = get_movie_details(movie_id)
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<int:movie_id>', methods=['GET'])
def analyze_movie(movie_id):
    """Analyze sentiment - show one positive and one negative review"""
    try:
        # Get movie details
        movie = get_movie_details(movie_id)
        print("\n" + "="*60)
        print(f"🎬 Analyzing: {movie.get('title')}")
        print("="*60)
        
        # Get reviews from TMDB with full data
        reviews_data = get_movie_reviews(movie_id)
        print(f"📊 Found {len(reviews_data)} reviews from TMDB")
        
        # If no reviews available
        if not reviews_data or len(reviews_data) == 0:
            print("❌ No reviews available for analysis")
            print("="*60 + "\n")
            return jsonify({
                'success': False,
                'movie': {
                    'id': movie['id'],
                    'title': movie['title'],
                    'poster_path': movie.get('poster_path'),
                    'rating': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count')
                },
                'message': 'No reviews available for this movie on TMDB.',
                'suggestion': 'Try popular movies like "The Shawshank Redemption" (id: 278), "Inception" (id: 27205), or "The Dark Knight" (id: 155)'
            })
        
        # Find best positive and negative examples
        best_positive, best_negative = analyzer.find_positive_and_negative_examples(reviews_data)
        
        # Calculate overall statistics
        review_texts = [r['content'] for r in reviews_data]
        overall_stats = analyzer.analyze_batch(review_texts)
        
        # Print sentiment analysis results to console
        print("\n📈 SENTIMENT ANALYSIS RESULTS:")
        print("-" * 60)
        print(f"Total Reviews Analyzed: {overall_stats['total_reviews']}")
        print(f"Positive Reviews: {overall_stats['positive_count']} ({overall_stats['positive_percentage']}%)")
        print(f"Negative Reviews: {overall_stats['negative_count']} ({overall_stats['negative_percentage']}%)")
        print(f"Average Confidence: {overall_stats['average_confidence']}")
        
        # Determine overall sentiment
        if overall_stats['positive_percentage'] > overall_stats['negative_percentage']:
            overall_sentiment = "POSITIVE ✓"
            emoji = "😊"
        elif overall_stats['negative_percentage'] > overall_stats['positive_percentage']:
            overall_sentiment = "NEGATIVE ✗"
            emoji = "😞"
        else:
            overall_sentiment = "NEUTRAL ⚖"
            emoji = "😐"
        
        print(f"\n{emoji} OVERALL SENTIMENT: {overall_sentiment}")
        print(f"   → {overall_stats['positive_percentage']}% Positive vs {overall_stats['negative_percentage']}% Negative")
        print("="*60 + "\n")
        
        # Build response
        response = {
            'success': True,
            'movie': {
                'id': movie['id'],
                'title': movie['title'],
                'poster_path': movie.get('poster_path'),
                'rating': movie.get('vote_average'),
                'vote_count': movie.get('vote_count'),
                'release_date': movie.get('release_date'),
                'overview': movie.get('overview')
            },
            'sentiment_summary': {
                'total_reviews_analyzed': overall_stats['total_reviews'],
                'positive_count': overall_stats['positive_count'],
                'negative_count': overall_stats['negative_count'],
                'positive_percentage': overall_stats['positive_percentage'],
                'negative_percentage': overall_stats['negative_percentage'],
                'average_confidence': overall_stats['average_confidence'],
                'overall_sentiment': overall_sentiment.replace(' ✓', '').replace(' ✗', '').replace(' ⚖', '')
            },
            'example_positive_review': best_positive if best_positive else {
                'message': 'No positive reviews found',
                'content': None
            },
            'example_negative_review': best_negative if best_negative else {
                'message': 'No negative reviews found', 
                'content': None
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze sentiment of custom text"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text field required'}), 400
    
    text = data['text']
    
    if isinstance(text, str):
        result = analyzer.analyze_single(text)
        return jsonify(result)
    elif isinstance(text, list):
        results = analyzer.analyze_batch(text)
        return jsonify(results)
    else:
        return jsonify({'error': 'Text must be string or array'}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask Backend Server")
    print("="*60)
    print("Server running at: http://localhost:5000")
    print("API Endpoints:")
    print("  - GET  /api/health")
    print("  - GET  /api/search?query=<movie_name>")
    print("  - GET  /api/movie/<movie_id>")
    print("  - GET  /api/analyze/<movie_id>")
    print("  - POST /api/analyze/text")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)