def apply_health_rules(recipes_df, user_health_conditions):
    if not user_health_conditions:
        return recipes_df
    
    filtered = recipes_df.copy()
    
    for condition in [c.lower() for c in user_health_conditions]:
        if "heart" in condition or "cholesterol" in condition:
            filtered = filtered[~filtered['health_recommendation'].str.contains('heart|cholesterol|oil|ghee', case=False, na=False)]
        elif "diabetes" in condition or "sugar" in condition:
            filtered = filtered[~filtered['ingredients'].str.contains('sugar|rice|bread|maida', case=False, na=False)]
        elif "hypertension" in condition or "blood pressure" in condition:
            filtered = filtered[~filtered['ingredients'].str.contains('salt|cheese|butter', case=False, na=False)]
    
    return filtered