import os
import re
import pandas as pd
from fuzzywuzzy import fuzz

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "food", "final dataset.csv")

df = pd.read_csv(DATA_CSV).fillna("")

def split_ingredients(text):
    return [i.strip().lower() for i in text.split(",") if i.strip()]

def find_by_ingredients(user_ingredients):
    user_list = split_ingredients(user_ingredients)
    matches = []

    for _, row in df.iterrows():
        recipe_ings = split_ingredients(row["ingredients"])
        score = sum(1 for i in user_list if i in " ".join(recipe_ings))
        if score >= max(1, len(user_list) // 2):
            matches.append(row.to_dict())

    return matches[:12]

def find_by_name(name):
    name = name.lower().strip()
    best_score = 0
    best_row = None

    for _, row in df.iterrows():
        score = fuzz.token_set_ratio(name, row["recipe_name"].lower())
        if score > best_score:
            best_score = score
            best_row = row

    return best_row.to_dict() if best_score > 65 else None

def clean_steps(text):
    # Split into steps by Step keyword or full stops
    raw_steps = re.split(r"(?:Step\s*\d+:|\n|\.)", text)

    cleaned = []
    for step in raw_steps:
        step = step.strip()

        # Remove any leading numbers like "1:", "2.", "3)", "4 -"
        step = re.sub(r"^\s*\d+\s*[:.)-]\s*", "", step)

        if len(step) > 10:
            cleaned.append(step)

    return cleaned
