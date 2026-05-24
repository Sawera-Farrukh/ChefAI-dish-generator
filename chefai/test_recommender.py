from utils.recommender import RecipeRecommender

def test():
    print("Testing Recipe Recommender...\n")
    
    recommender = RecipeRecommender()
    
    user_input = "chicken tomato ginger garlic chili"
    
    print(f"User Ingredients: {user_input}\n")
    print("Finding similar recipes...\n")
    
    results = recommender.get_similar_recipes(user_input, top_n=5)
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{i}. {row['recipe_name']}")
        print(f"   Similarity: {row['similarity_score']:.4f}")
        print(f"   Cuisine: {row.get('cuisine_path', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    test()