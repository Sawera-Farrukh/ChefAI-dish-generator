from .recommender import RecipeRecommender
from .rules import apply_health_rules

class SimpleMLManager:
    def __init__(self):
        self.recommender = RecipeRecommender()   # This will now load your real dataset

    def recommend(self, ingredients: str, health_conditions=None, top_n=6):
        # TF-IDF based similar recipes
        results = self.recommender.get_similar_recipes(ingredients, top_n=top_n*2)
        
        # Apply health rules if any
        if health_conditions and len(health_conditions) > 0:
            results = apply_health_rules(results, health_conditions)
        
        # Optional: Re-rank using Decision Tree confidence
        if not results.empty:
            results['ml_score'] = 0.0
            for idx in results.index:
                try:
                    _, prob = self.recommender.predict_with_ml(ingredients, model_type="decision_tree")
                    results.at[idx, 'ml_score'] = prob
                except:
                    pass
            
            results = results.sort_values(by=['similarity_score', 'ml_score'], ascending=False)
        
        return results.head(top_n)