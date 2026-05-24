import pandas as pd
import os
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Add project root to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recommender import RecipeRecommender

def train_simple_ml_models():
    """Train TF-IDF + Decision Tree + Naive Bayes"""
    print("🚀 Training Simple ML Models (TF-IDF + DT + NB)...\n")
    
    recommender = RecipeRecommender()
    recommender.train_tfidf()
    recommender.train_ml_models()
    
    print("✅ Simple ML Models trained successfully!\n")

def train_gpt2_model():
    """Fine-tune GPT-2 on your recipe dataset"""
    print("🔥 Starting GPT-2 Fine-tuning...\n")
    
    # -----------------------------
    # 1) Load CSV dataset
    # -----------------------------
    csv_path = "data/food/final dataset.csv"        # ← Fixed path (relative to project root)
    print(f"Loading dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["recipe_name", "ingredients", "directions"])
    print(f"✅ Loaded {len(df)} recipes.")

    # -----------------------------
    # 2) Build training text
    # -----------------------------
    def make_text(row):
        return f"""Recipe Name: {row['recipe_name']}

Ingredients:
{row['ingredients']}

Directions:
{row['directions']}

Nutrition:
{row.get('nutrition', 'N/A')}

Cuisine:
{row.get('cuisine_path', 'N/A')}
"""

    df["text"] = df.apply(make_text, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

    # -----------------------------
    # 3) Load GPT-2
    # -----------------------------
    print("Loading GPT-2 tokenizer & model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # -----------------------------
    # 4) Tokenization
    # -----------------------------
    def tokenize(batch):
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_ds = dataset.map(tokenize, batched=True)

    # -----------------------------
    # 5) Training Arguments (Fast & Light)
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="models/chefai-gpt2-fast",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,           # Increased a bit
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        warmup_steps=50,
        learning_rate=5e-5,
        fp16=False,                              # Set True if you have GPU
        report_to="none",
    )

    # -----------------------------
    # 6) Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )

    # -----------------------------
    # 7) Train
    # -----------------------------
    print("\n🔥 Starting GPT-2 Fine-tuning...")
    trainer.train()

    # -----------------------------
    # 8) Save Model
    # -----------------------------
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model("models/chefai-gpt2-fast")
    
    print("\n✅ GPT-2 Fine-tuning Completed Successfully!")


if __name__ == "__main__":
    print("="*60)
    print("CHEF AI - TRAINING SCRIPT")
    print("="*60)
    
    # First train simple ML models (TF-IDF, Decision Tree, Naive Bayes)
    train_simple_ml_models()
    
    # Then fine-tune GPT-2 (this takes longer)
    train_gpt2_model()
    
    print("\n🎉 ALL TRAINING COMPLETED!")