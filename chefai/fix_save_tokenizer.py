from transformers import GPT2Tokenizer

model_path = "models/chefai-gpt2-fast"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(model_path)

print("Tokenizer files saved successfully!")
