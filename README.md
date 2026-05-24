# 🍳 ChefAI - AI-Powered Recipe Discovery & Recommendation

**ChefAI** is an intelligent recipe discovery platform that uses machine learning to recommend recipes based on available ingredients or dish names. Built with Flask and powered by advanced recommendation algorithms, ChefAI helps users find perfect recipes tailored to their needs.

## ✨ Features

- 🔍 **Search by Dish Name** - Find recipes by entering your favorite dish name using fuzzy matching
- 🥘 **Search by Ingredients** - Discover recipes based on ingredients you have on hand
- 🤖 **ML-Powered Recommendations** - Uses TF-IDF vectorization and cosine similarity for intelligent recipe matching
- 👤 **User Authentication** - Secure signup and login system with encrypted password storage
- 📊 **Multiple ML Models** - Decision Tree and Naive Bayes classifiers for cuisine prediction
- 🎯 **Smart Matching** - Fuzzy string matching for better recipe name recognition
- 📱 **Responsive UI** - Clean, intuitive web interface with recipe details

## 🚀 Getting Started

### Prerequisites
- Python 3.10
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
cd chefai-dish-generator
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize the database**
```bash
python init_db.py
```

## 💻 Running the Application

### Start the Flask Development Server
```bash
python app.py
```

The application will run at: `http://localhost:5000`

### First Time Setup

1. Visit the home page
2. Click **Sign Up** to create an account
3. Log in with your credentials
4. Start searching for recipes!

## 🔧 Core Features & How They Work

### 1. **Recipe Search by Dish Name**
- Uses fuzzy string matching (fuzz.token_set_ratio)
- Tolerance level: scores above 65 are considered matches
- Finds the best matching recipe from the dataset

### 2. **Recipe Search by Ingredients**
- Splits user input into individual ingredients
- Matches ingredients against recipe database
- Returns top 12 recipes with highest ingredient overlap
- Minimum match threshold: 50% of searched ingredients must be found

### 3. **ML-Based Recommendations**
- **TF-IDF Vectorization**: Converts ingredient text to numeric vectors
- **Cosine Similarity**: Measures similarity between user ingredients and recipes
- **Cuisine Prediction**: Predicts likely cuisine using trained ML models
- **Feature Engineering**: Captures multi-word ingredients (n-grams) like "chicken karahi"

### 4. **User Authentication**
- Secure password hashing using Werkzeug
- Session-based authentication with Flask
- SQLite database for user storage
- Prevents duplicate usernames

## 📚 Key Modules

### `app.py` - Flask Application
Main entry point with routes for:
- `GET /` - Home page
- `GET/POST /signup` - User registration
- `GET/POST /login` - User login
- `GET /logout` - Session termination
- `POST /generate` - Recipe search and generation

### `utils/recipe_generator.py`
**Functions:**
- `find_by_name(name)` - Search recipe by dish name
- `find_by_ingredients(ingredients)` - Search by available ingredients
- `clean_steps(text)` - Parse and clean recipe instructions
- `split_ingredients(text)` - Parse ingredient strings

### `utils/recommender.py`
**RecipeRecommender Class:**
- `load_data()` - Load recipe dataset
- `train_tfidf()` - Train TF-IDF vectorizer
- `get_similar_recipes(ingredients, top_n)` - Find similar recipes
- `train_ml_models()` - Train Decision Tree & Naive Bayes
- `predict_with_ml(ingredients, model_type)` - Predict cuisine

### `init_db.py`
Creates SQLite database with users table:
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)
```

## 🎯 Usage Examples

This dataset is provide in data/food/final dataset.csv 

### Example 1: Search by Dish Name
```
Enter: "Biryani"
Result: Best matching biryani recipe with ingredients and directions
```

### Example 2: Search by Ingredients
```
Enter: "chicken, onion, ginger, garlic"
Result: Top 12 recipes that use these ingredients
```

