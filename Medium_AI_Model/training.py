#!/usr/bin/env python3
import numpy
import torch
import math
from math import e
import os
import json

weights = torch.load("Medium_AI_Model/model.pt", weights_only=True)
QM = weights["QM"] 
KW = weights["KW"] 
VW = weights["VW"] 
EM = weights["EM"] 
shape_matrix = weights["shape_matrix"]
deshape_matrix = weights["deshape_matrix"] 
gamma = weights["gamma"] 
beta = weights["beta"] 
output = weights["output"]

parameters = [
    QM,
    KW,
    VW,
    EM,
    shape_matrix,
    deshape_matrix,
    gamma,
    beta,
    output
]
for parameter in parameters:
    parameter.requires_grad_(True)
def data():
    with open("Medium_AI_Model/tokenizer.json", "r") as file:
        tokens = json.load(file)
        vocabulary = tokens["vocabulary"]
        return vocabulary

def get_target_text():
    books = []

    with open("Medium_AI_Model/training_text.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            books.append(data["text"])
            

    books = "\n\n".join(books)

    byte_data = books.encode("utf-8")
    return list(byte_data)  

def numberify():
    prompt_list_index = 0
    with open("Medium_AI_Model/tokenizer.json", "r") as file:
        embedded_text = []
        prompt_list = []
        tokenized_list = []
        data = json.load(file)
        tokens = data["vocabulary"]
        merge_rules = data["merge_rules"]
    prompt = get_target_text()
    for letter in range(len(prompt)):
        prompt_list.append(prompt[letter])
    while prompt_list_index < len(prompt_list):
        match_happened = 0
        if prompt_list_index == len(prompt_list) - 1:
            tokenized_list.append(prompt_list[prompt_list_index])
            break
        else:
            prompt_list_pair = [prompt_list[prompt_list_index], prompt_list[prompt_list_index + 1]]
        for index, merge_rule in enumerate(merge_rules):
            
            if prompt_list_pair == merge_rule[0]:
                prompt_list[prompt_list_index] = merge_rule[1]
                prompt_list.pop(prompt_list_index + 1)
                match_happened = 1   
                break 
                
        if match_happened == 0:
            tokenized_list.append(prompt_list[prompt_list_index])
            prompt_list_index +=1
    tokenized_list = prompt_list.copy()
    return tokenized_list


def embedding_matrix(embedded_text):
    embedding_matrix = []
    for embed in embedded_text:
        embedding_matrix.append(EM[embed])
    embedding_matrix = torch.stack(embedding_matrix)
    return embedding_matrix

def layer_norm(x):
    orginal_format = x.shape
    epsilon = 1e-9
    total_list = []
    for row in x:
        variance_mean = 0
        mean = row.mean()
        centered_values_list = []
        variance_list = []
        for weight in row:
            weight_value = weight - mean
            centered_values_list.append(weight_value)
            weight_value_mean = weight_value **2
            variance_list.append(weight_value_mean) 
        for number in variance_list:
            variance_mean = number + variance_mean
        variance = variance_mean / len(variance_list)
        variance = torch.sqrt(variance + epsilon)
        total = torch.stack(centered_values_list)
        total = total / variance
        total *= gamma
        total += beta
        total_list.append(total)
        final_value = torch.stack(total_list)
        value_mean = sum(total) / len(total)
    return final_value.reshape(orginal_format)



def attention(ET): #Embedded Text
    final_values = []
    ET = embedding_matrix(ET)
    orginal_format = ET.shape
    Q = ET @ QM # query value
    K = ET @ KW # Key Value
    V = ET @ VW # Value Value
    # Transposing and scaling the numbers
    scores = Q @ K.transpose(-2, -1)
    scores = scores / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
    casual_scores = casual_mask(scores)
    casual_scores = softmax(casual_scores)
    print(casual_scores)
    for query in range(orginal_format[0]):
        weights = casual_scores[query]
        weights = weights.unsqueeze(-1)
        value_weights = weights * V
        final_value = torch.sum(value_weights, dim=0)
        final_values.append(final_value)
    final_values = torch.stack(final_values)
    final_values = final_values.reshape(orginal_format)
    final_values = ET + final_values
    return layer_norm(final_values)
        

# I don't need it for training
def decode(final_numbers):
    output = []
    with open("tokenizer.json", "r") as file:
        data = json.load(file)
        tokens = data["vocabulary"]
    for number in embedded_text:
        for token in tokens:
            token_number = tokens[token]
            if number == token_number:
                output.append(token)
    response = "".join(output)
    print(response)

def casual_mask(embedded_text):
    casual_mask = []
    sequence_length = embedded_text.shape[0]
    embedding_size = embedded_text.shape[1]
    for query_index, query in enumerate(embedded_text):
        for key_index, key in enumerate(query):
            if query_index < key_index:
                key = torch.tensor(float("-inf"))
            casual_mask.append(key)
    casual_mask = torch.stack(casual_mask)
    casual_mask = casual_mask.reshape(sequence_length, embedding_size)
    return casual_mask

def softmax(x):
    orginal_format = x.shape
    combined_numbers= 0
    numbers = []
    final_numbers = []
    for row in x: # the second 8 and the last one is each weight
        biggest_number = -9999999999
        for number in row:
            if number > biggest_number:
                biggest_number = number

        normalized_row = row - biggest_number #biggest number is used to prevent inf
        
        power_of_e = e ** normalized_row
        
        for i in power_of_e:
            final_numbers.append(i / sum(power_of_e))
            
    final_numbers = torch.stack(final_numbers)
    final_numbers = torch.reshape(final_numbers, orginal_format)
    return final_numbers

def FFN(x):
    expanded_matrix = x @ shape_matrix
    gelu_output = gelu(expanded_matrix)
    normal_dimensions = gelu_output @ deshape_matrix
    normalized_dimensions = layer_norm(normal_dimensions)  
    return normalized_dimensions + x

def gelu(x):
    gelu = (x * 0.5) * (1 + torch.tanh((x + 0.044715 * x ** 3) * torch.sqrt(torch.tensor(2 / torch.pi))))
    return gelu

def parameter_update(func_output):
    learning_rate = 0.02
    loss = cross_entropy(func_output)   
    loss.backward()
    with torch.no_grad():
        for parameter in parameters:
            parameter -= learning_rate * parameter.grad
            parameter.grad.zero_()
    return loss

def cross_entropy(x):
    x = x @ output
    x = softmax(x)
    total_probablities = []
    for row_index in range(len(x) - 1):
            correct_token = embedded_text[row_index + 1]
            selected_probablity = x[row_index, correct_token]
            selected_probablity = -torch.log(selected_probablity)
            total_probablities.append(selected_probablity)

    mean = sum(total_probablities) / len(total_probablities)
    return mean

embedded_text = numberify()
x = attention(embedded_text)
y = FFN(x)
a = parameter_update(y)
#decode(embedded_text)


torch.save({
    "QM": QM,
    "KW": KW,
    "VW": VW,
    "EM": EM,
    "shape_matrix": shape_matrix,
    "deshape_matrix": deshape_matrix,
    "gamma": gamma,
    "beta": beta,
    "output": output
}, "model.pt")