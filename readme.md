
# 🎬 Movie Review Sentiment Analysis

A full-stack web application that analyzes the sentiment of movie reviews using machine learning. Users can search for movies and get AI-powered sentiment analysis showing positive and negative review examples.

## 📋 Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- 🔍 **Movie Search**: Search for any movie using The Movie Database (TMDB) API
- 🤖 **AI Sentiment Analysis**: Analyze movie reviews using VADER sentiment analysis
- 📊 **Statistics Dashboard**: View overall positive/negative percentages
- 💬 **Review Examples**: See the most confident positive and negative review examples
- 🎯 **Real-time Analysis**: Instant sentiment scoring with confidence levels
- 📈 **Detailed Metrics**: Average confidence scores and review breakdowns

## 🛠 Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **VADER Sentiment** - Sentiment analysis model
- **TMDB API** - Movie data source
- **Flask-CORS** - Cross-origin resource sharing

### Machine Learning
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)** - Pre-trained sentiment analysis tool optimized for social media and review text

## 📁 Project Structure

```
backend/
├── app.py                      # Main Flask application
├── sentiment_analyzer.py       # Sentiment analysis logic
├── model_trainer.py           # Model creation script
├── vader_model.py             # VADER model wrapper
├── tmdb_api.py                # TMDB API integration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys)
├── README.md                  # This file
├── models/
│   └── sentiment_model.pkl    # Trained sentiment model
└── venv/                      # Virtual environment (not in git)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A TMDB API key (free - see Configuration section)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd Sentiment-Analysis/backend
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create the Sentiment Model
```bash
python model_trainer.py
```

You should see output like:
```
Creating VADER sentiment analyzer...
✓ Model saved to models/sentiment_model.pkl
Test predictions:
...
```

## ⚙️ Configuration

### Get TMDB API Key

1. Go to [The Movie Database](https://www.themoviedb.org/)
2. Create a free account
3. Navigate to Settings → API
4. Request an API key (choose "Developer" option)
5. Copy your API key

### Create .env File

Create a file named `.env` in the `backend/` folder:

```env
TMDB_API_KEY=your_actual_api_key_here
```

**Important**: Never commit your `.env` file to git! It's already in `.gitignore`.

## 🏃 Running the Application

### Start the Backend Server

```bash
# Make sure you're in the backend folder with venv activated
python app.py
```

You should see:
```
============================================================
🚀 Starting Flask Backend Server
============================================================
Server running at: http://localhost:5000
API Endpoints:
  - GET  /api/health
  - GET  /api/search?query=<movie_name>
  - GET  /api/movie/<movie_id>
  - GET  /api/analyze/<movie_id>
  - POST /api/analyze/text
============================================================
```

### Test the Server

Open your browser and visit:
- **Health Check**: http://localhost:5000/api/health
- **Search**: http://localhost:5000/api/search?query=inception
- **Analyze**: http://localhost:5000/api/analyze/278

## 📡 API Endpoints

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "message": "Backend is running"
}
```

### 2. Search Movies
```http
GET /api/search?query=<movie_name>
```
**Example:**
```http
GET /api/search?query=inception
```
**Response:**
```json
{
  "results": [
    {
      "id": 27205,
      "title": "Inception",
      "release_date": "2010-07-16",
      "poster_path": "/...",
      "overview": "..."
    }
  ],
  "total": 1
}
```

### 3. Get Movie Details
```http
GET /api/movie/<movie_id>
```
**Example:**
```http
GET /api/movie/27205
```

### 4. Analyze Movie Sentiment
```http
GET /api/analyze/<movie_id>
```
**Example:**
```http
GET /api/analyze/278
```
**Response:**
```json
{
  "success": true,
  "movie": {
    "id": 278,
    "title": "The Shawshank Redemption",
    "poster_path": "/...",
    "rating": 8.7,
    "vote_count": 23450
  },
  "sentiment_summary": {
    "total_reviews_analyzed": 5,
    "positive_count": 4,
    "negative_count": 1,
    "positive_percentage": 80.0,
    "negative_percentage": 20.0,
    "average_confidence": 0.85,
    "overall_sentiment": "POSITIVE"
  },
  "example_positive_review": {
    "author": "John Doe",
    "content": "This movie is absolutely amazing!...",
    "sentiment": "positive",
    "confidence": 0.92,
    "positive_probability": 0.92,
    "negative_probability": 0.08
  },
  "example_negative_review": {
    "author": "Jane Smith",
    "content": "I didn't enjoy this film...",
    "sentiment": "negative",
    "confidence": 0.78
  }
}
```

### 5. Analyze Custom Text
```http
POST /api/analyze/text
Content-Type: application/json

{
  "text": "This movie was absolutely amazing!"
}
```
**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.91,
  "positive_probability": 0.91,
  "negative_probability": 0.09
}
```

## 🔍 How It Works

### 1. User Searches for a Movie
- Frontend sends search query to `/api/search`
- Backend queries TMDB API
- Returns list of matching movies

### 2. User Selects a Movie to Analyze
- Frontend sends movie ID to `/api/analyze/<movie_id>`
- Backend fetches movie details and reviews from TMDB
- Reviews are passed through VADER sentiment analyzer

### 3. Sentiment Analysis Process
```python
# VADER analyzes each review
for review in reviews:
    scores = analyzer.polarity_scores(review)
    # scores = {
    #   'pos': 0.8,      # Positive score
    #   'neg': 0.1,      # Negative score
    #   'neu': 0.1,      # Neutral score
    #   'compound': 0.7  # Overall score (-1 to 1)
    # }
```

### 4. Results Aggregation
- Identifies best positive and negative examples
- Calculates overall percentages
- Returns structured results to frontend

## 🧪 Testing

### Test with PowerShell
```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:5000/api/health"

# Search
Invoke-WebRequest -Uri "http://localhost:5000/api/search?query=avatar"

# Analyze
Invoke-WebRequest -Uri "http://localhost:5000/api/analyze/278"

# Custom text analysis
$body = @{text = "This movie was great!"} | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:5000/api/analyze/text" -Method POST -Body $body -ContentType "application/json"
```

### Test with curl
```bash
# Health check
curl http://localhost:5000/api/health

# Search
curl "http://localhost:5000/api/search?query=inception"

# Analyze
curl http://localhost:5000/api/analyze/278

# Custom text
curl -X POST http://localhost:5000/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

### Movies with Reviews (Good for Testing)
- The Shawshank Redemption: `278`
- Inception: `27205`
- The Dark Knight: `155`
- Interstellar: `157336`
- Fight Club: `550`

## 🐛 Troubleshooting

### Virtual Environment Issues (Windows)
```powershell
# If you get execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

### Module Not Found Errors
```bash
# Make sure venv is activated (you should see (venv) in terminal)
# Then reinstall dependencies
pip install -r requirements.txt
```

### TMDB API Errors
```bash
# Check your .env file exists and has correct API key
cat .env  # Mac/Linux
type .env  # Windows

# Verify the key works
curl "https://api.themoviedb.org/3/movie/278?api_key=YOUR_KEY"
```

### No Reviews Found
Most movies on TMDB have very few reviews. Try these movies that typically have reviews:
- The Shawshank Redemption (278)
- The Dark Knight (155)
- Inception (27205)

### Port Already in Use
```bash
# If port 5000 is busy, change it in app.py:
app.run(debug=True, port=5001)  # Use different port
```

## 🎯 Future Enhancements

- [ ] Add React frontend
- [ ] Support multiple review sources (IMDb, Rotten Tomatoes)
- [ ] Add review filtering (by date, rating)
- [ ] Implement caching for faster responses
- [ ] Add user authentication
- [ ] Export analysis results as PDF
- [ ] Visualize sentiment trends over time
- [ ] Support batch movie analysis
- [ ] Add sentiment comparison between movies

## 📝 Notes

- **TMDB Review Limitation**: Most movies have few or no reviews on TMDB. This is normal - TMDB is primarily a movie database, not a review platform.
- **VADER Model**: Pre-trained and optimized for social media/review text. No training data needed!
- **Rate Limits**: TMDB API has rate limits. Be mindful when making many requests.

## 📄 License

This project is for educational purposes.

## 👥 Contributors

Your Name - Initial work

## 🙏 Acknowledgments

- [TMDB API](https://www.themoviedb.org/documentation/api) for movie data
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- Flask documentation and community
```


