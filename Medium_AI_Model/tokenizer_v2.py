#!/usr/bin/env python3
import json
def get_data():
    books = []

    with open("Medium_AI_Model/training_text.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            books.append(data["text"])
            break
    books = "\n\n".join(books)

    byte_data = books.encode("utf-8")
    return list(byte_data)  

def get_pairs():
    reverse_lookup = {}
    token_number = 256
    vocabulary = {}
    tokens = {}
    merge_rules = []
    bytes = get_data()
    bytes_index = 0
    for qwerty in range(300):
        bytes_index = 0
        pairs = {}
        most_matches = 0
        while bytes_index < len(bytes) - 1:
            pair = bytes[bytes_index], bytes[bytes_index + 1]
            if pair not in pairs:
                pairs[pair] = 1
            else: 
                pairs[pair] +=1
            bytes_index += 1

        bytes_index = 0
        for pair, count in pairs.items():
            if pairs[pair] > most_matches:
                most_matches = pairs[pair]
                match = pair
        tokens[match] = token_number
        
        while bytes_index < len(bytes) - 1:
            pair = bytes[bytes_index], bytes[bytes_index + 1]
            if pair == match:
                bytes[bytes_index] = match
                bytes.pop(bytes_index + 1)
            else:
                bytes_index +=1
        id_match = list(match)
        for tuples_index, tuples in enumerate(match):
            if tuples in reverse_lookup:
                id_match[tuples_index] = reverse_lookup[tuples]

        vocabulary[str(token_number)] = match
        match = tuple(match)
        id_match = tuple(id_match)
        reverse_lookup[match] = token_number
        merge_rules_appending = [id_match, token_number]
        merge_rules.append(merge_rules_appending)
        
        
        token_number += 1
    
    with open("Medium_AI_Model/tokenizer.json", "w") as file:
        upload_data = {
            "vocabulary":vocabulary,
            "merge_rules":merge_rules
        }
        json.dump(upload_data, file, indent = 2)
get_pairs()


        
        