import json
from datasets import load_dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
text = {"text" : dataset}
#with open("Medium_AI_Model/raw.txt", "r") as file:
    #text = file.read()
#text = {"text": text}
with open("Medium_AI_Model/training_text.jsonl", "a") as file:
    json.dump(text, file)
    file.write("\n")