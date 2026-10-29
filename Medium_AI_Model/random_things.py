import json
import heapq
def numberify():
    global embeded_matrix
    prompt_list_index = 0
    with open("Medium_AI_Model/tokenizer.json", "r") as file:
        prompt_list = []
        tokenized_list = []
        data = json.load(file)
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
            prompt_list_index += 1
    tokenized_list = prompt_list.copy()
    embeded_matrix = tokenized_list
    return tokenized_list

def new_numerify():
    heap = []
    tokenized_list = []
    current = 0
    prompt_list_index = 0
    python_merge_rules = {}
    with open("Medium_AI_Model/tokenizer.json", "r") as file:
        #Init arrays
        prompt_list = get_target_text()
        next = list(range(1, len(prompt_list))) + [-1]
        previous = [-1] + list(range(len(prompt_list) - 1))
        alive = [True for n in prompt_list]
        
        data = json.load(file)
        merge_rules = data["merge_rules"]
        for index, pair in enumerate(merge_rules):
            python_merge_rules[tuple(merge_rules[index][0])] = merge_rules[index][1]

    while prompt_list_index != -1:
        if next[prompt_list_index] != -1:
            pair = (prompt_list[prompt_list_index],prompt_list[next[prompt_list_index]])
            merged_token = python_merge_rules.get(pair)

            if merged_token is not None:
                heapq.heappush(heap,(merged_token, prompt_list_index, next[prompt_list_index]))

        prompt_list_index = next[prompt_list_index]

    while len(heap) > 0:
        token = heapq.heappop(heap)
        if not alive[token[1]] or not alive[token[2]]:
                continue
        else:
            if next[token[1]] == token[2]: 
                current_next = next[token[1]]
                next_next = next[current_next]
                prompt_list[token[1]] = token[0]
                alive[current_next] = False
                next[token[1]] = next[current_next]
                if next_next != -1:
                    previous[next_next] = token[1]
                if next[token[1]] != -1:
                    try_token = python_merge_rules.get((prompt_list[token[1]], prompt_list[next[token[1]]]))
                    if try_token != None:
                        heapq.heappush(heap, (try_token, token[1], next[token[1]]))
                if previous[token[1]] != -1:
                    try_token = python_merge_rules.get((prompt_list[previous[token[1]]], prompt_list[token[1]]))
                    if try_token != None:
                        heapq.heappush(heap, (try_token, previous[token[1]], token[1]))
            else:
                continue

    while current != -1:
        tokenized_list.append(prompt_list[current])
        current = next[current]
    print(tokenized_list)


    while current != -1:
        tokenized_list.append(prompt_list[current])
        current = next[current] 
            
new_numerify()