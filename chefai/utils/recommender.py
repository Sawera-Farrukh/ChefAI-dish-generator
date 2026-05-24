import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

class RecipeRecommender:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.dt_model = None
        self.nb_model = None
        self.recipes_df = None
        self.load_data()

    def load_data(self):
        dataset_path = "data/food/final dataset.csv"
        
        if os.path.exists(dataset_path):
            print(f"✅ Loading dataset: {dataset_path}")
            self.recipes_df = pd.read_csv(dataset_path)
            
            # Clean and prepare ingredients for TF-IDF
            self.recipes_df['ingredients_str'] = self.recipes_df['ingredients'].astype(str)
            
            # Optional: Clean cuisine_path for better use
            self.recipes_df['cuisine'] = self.recipes_df['cuisine_path'].str.split('/').str[-1]
            
            print(f"✅ Dataset loaded! Total recipes: {len(self.recipes_df)}")
        else:
            raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

    def train_tfidf(self):
        """Train TF-IDF on ingredients"""
        print("🔄 Training TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=8000,
            ngram_range=(1, 2)   # captures "chicken karahi", "ginger garlic" etc.
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['ingredients_str'])
        
        os.makedirs("models", exist_ok=True)
        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        print("✅ TF-IDF model trained and saved successfully.")

    def get_similar_recipes(self, user_ingredients: str, top_n=6):
        """Find recipes with similar ingredients using TF-IDF"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            self.train_tfidf()

        user_vec = self.vectorizer.transform([user_ingredients])
        similarity_scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        
        # Get top similar recipes
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        results = self.recipes_df.iloc[top_indices].copy()
        results['similarity_score'] = similarity_scores[top_indices]
        
        return results[['recipe_name', 'image_url', 'ingredients', 'directions', 
                        'cuisine_path', 'nutrition', 'health_recommendation', 
                        'similarity_score']]

    def train_ml_models(self):
        """Train Decision Tree and Naive Bayes"""
        print("🔄 Training Machine Learning models...")
        
        X = self.tfidf_matrix
        y = self.recipes_df['cuisine']   # Predicting cuisine as target (you can change this)

        self.dt_model = DecisionTreeClassifier(max_depth=12, random_state=42)
        self.nb_model = MultinomialNB()

        self.dt_model.fit(X, y)
        self.nb_model.fit(X, y)

        # Save models
        with open("models/decision_tree.pkl", "wb") as f:
            pickle.dump(self.dt_model, f)
        with open("models/naive_bayes.pkl", "wb") as f:
            pickle.dump(self.nb_model, f)
        
        print("✅ Decision Tree and Naive Bayes models trained and saved.")

    def predict_with_ml(self, user_ingredients: str, model_type="decision_tree"):
        """Predict using ML model (e.g. likely cuisine)"""
        if self.vectorizer is None:
            self.train_tfidf()

        user_vec = self.vectorizer.transform([user_ingredients])
        
        if model_type == "decision_tree" and self.dt_model:
            pred = self.dt_model.predict(user_vec)[0]
            prob = self.dt_model.predict_proba(user_vec).max()
        elif model_type == "naive_bayes" and self.nb_model:
            pred = self.nb_model.predict(user_vec)[0]
            prob = self.nb_model.predict_proba(user_vec).max()
        else:
            pred, prob = "Unknown", 0.0

        return pred, round(prob, 3)