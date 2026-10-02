import json

with open("Medium_AI_Model/raw.txt", "r") as file:
    text = file.read()
text = {"text": text}
with open("Medium_AI_Model/training_text.jsonl", "a") as file:
    json.dump(text, file)
    file.write("\n")