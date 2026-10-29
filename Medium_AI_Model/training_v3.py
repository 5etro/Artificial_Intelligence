import numpy as np
import json
import heapq
global start, done, vocabulary
done = False
start = 0
learning_rate = 0.0001
standard_weights = 32
number_tokens = 0
#Weights (Will be moved to another file)
    # Token N Probablity weights

    # Attention Weights
QM = np.random.randn(standard_weights, standard_weights)
KM = np.random.randn(standard_weights, standard_weights)
VM = np.random.randn(standard_weights, standard_weights)
    #Layer Norm
gamma_ffn = np.ones(standard_weights)
beta_ffn = np.zeros(standard_weights)
gamma_atn = np.ones(standard_weights)
beta_atn = np.zeros(standard_weights)
    # FFN & GELU
expand_matrix = np.random.randn(standard_weights, standard_weights  * 2)
shrink_matrix = np.random.randn(standard_weights *2, standard_weights)

# Test Variables that will be deleted
    # BackProp

def data():
    global number_tokens, vocabulary 
    with open("Medium_AI_Model/tokenizer.json", "r") as file:
        tokens = json.load(file)
        vocabulary = tokens["vocabulary"]
        number_tokens = max(int(token) for token in vocabulary.keys()) + 1
        return vocabulary

def get_target_text():
    global byte_data
    books = []
    with open("Medium_AI_Model/training_text.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            books.append(data["text"])
    books = "\n\n".join(books)
    byte_data = books.encode("utf-8")
    byte_data = list(byte_data)
    return list(byte_data)

def numberify():
    global embeded_matrix
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
    embeded_matrix = tokenized_list
    return embeded_matrix

def embedding_matrix():
    global batch, start, done

    batch = embeded_matrix[start:start + 100]
    weighted_text = embed_matrix[batch]

    start += len(batch)

    if start >= len(embeded_matrix):
        done = True
    return weighted_text

def attention():
    global start, done
    EM = embedding_matrix()
    og_format = EM.shape
    Q = EM @ QM
    K = EM @ KM
    V = EM @ VM
    #Switches the second last dimension with the last dimension to allow K and Q to multiply together using there shared length in embedded text 
    scores = Q @ K.swapaxes(-2, -1)
    scores = scores / np.sqrt(K.shape[-1]) #Scaling Values
    #Casual Mask
    casual_mask = np.full((scores.shape), -np.inf)
    mask = np.triu(casual_mask, k=1)
    casual_scores = scores + mask 
    softmaxxed = softmax(casual_scores)
    values = softmaxxed @ V
    values_residual = values + EM
    final_values, deviation, variance, total_beta, total_gamma = layer_norm(values_residual, gamma_atn, beta_atn)
    return final_values, deviation, values_residual, V, values, softmaxxed, Q, K, EM, total_gamma, total_beta

def softmax(matrix):
    row_max = np.max(matrix, axis=1, keepdims=True)
    return (np.exp(matrix - row_max) / np.sum(np.exp(matrix - row_max), axis=1, keepdims=True))

def layer_norm(matrix, gamma, beta):
    epsilon = 1e-9
    centered_values = matrix - np.mean(matrix, axis=1, keepdims=True)
    variance = np.mean(centered_values **2, axis=1, keepdims=True)
    deviation = np.sqrt(variance + epsilon)
    total_gamma = centered_values / deviation
    total_beta = gamma * total_gamma
    total = beta + total_beta
    return total, deviation, variance, total_beta, total_gamma

def gelu(x):
    gelu = (x * 0.5) * (1 + np.tanh((x + 0.044715 * x ** 3) * np.sqrt(2 / np.pi)))
    return gelu

def FFN(matrix):
    expanded_matrix = matrix @ expand_matrix
    gelu_output = gelu(expanded_matrix)
    normal_dimensions = gelu_output @ shrink_matrix
    normed_dimensions, deviation, variance, total_beta, total_gamma = layer_norm(normal_dimensions, gamma_ffn, beta_ffn)
    return normed_dimensions + matrix, normed_dimensions, normal_dimensions, deviation, gelu_output, expanded_matrix, shrink_matrix, matrix, total_beta, total_gamma

def cross_entropy(matrix):
    logits = matrix @ output_matrix
    max_logit_row = np.max(logits, axis=1, keepdims=True)
    softmaxxed = softmax(logits)
    probablity = softmaxxed[np.arange(len(softmaxxed) -1 ), batch[1:]]
    probablities = ((np.log(np.sum(np.exp(logits[:-1]  - max_logit_row[:-1]), axis=1, keepdims=False))) + max_logit_row[:-1, 0]) - logits[np.arange(len(logits) -1), batch[1:]]
    return np.sum(probablities) / len(probablities), probablity, softmaxxed

def update():
    for index, parameter in enumerate(parameters):
        parameter = parameter[1]
        parameter -= learning_rate * grads[index]

def backprop(probability, token_probs, model_output,
             ffn_norm, ffn_pre_norm, ffn_std,
             gelu_out, ffn_pre_gelu, shrink_matrix, ffn_input,
             attn_std, attn_pre_norm, attn_norm,
             V, attn_values, attn_probs, Q, K, EM,
             pre_gamma_ffn_ln, pre_beta_ffn_ln, 
             pre_gamma_attn_ln, pre_beta_attn_ln):
    global grads
    logits_grad = np.zeros((len(batch), number_tokens))
    embed_matrix_grad = np.zeros((number_tokens, standard_weights))

    # Derivative of Cross Entropy
    #loss_grad = (token_probs[:-1] - 1) / len(probability)
    #prob_grad[np.arange(len(token_probs) - 1), embeded_matrix[1:]] = loss_grad

    # Derivative of Softmax / Logits
    logits_grad[:-1] = token_probs[:-1]
    logits_grad[np.arange(len(token_probs) - 1), batch[1:]] -= 1
    logits_grad /= len(token_probs) - 1


    # Derivative of Output Matrix
    output_matrix_grad = model_output.swapaxes(-2, -1) @ logits_grad

    # Derivative of FFN Output
    ffn_out_grad = logits_grad @ output_matrix.swapaxes(-2, -1)

    # FFN
    # Layer Norm
    ffn_ln_jacobian = layernorm_der(
        ffn_std, ffn_pre_norm, ffn_norm, gamma_ffn
    )
    ffn_ln_input_grad = np.sum(
        ffn_ln_jacobian * ffn_out_grad[:, :, None], axis=1
    )

    # Shrink Matrix
    shrink_matrix_grad = gelu_out.swapaxes(-2, -1) @ ffn_ln_input_grad
    gelu_out_grad = ffn_ln_input_grad @ shrink_matrix.swapaxes(-2, -1)

    # GELU
    pre_gelu_grad = (
        0.5 * (
            1 + np.tanh(
                np.sqrt(2 / np.pi) *
                (ffn_pre_gelu + 0.044715 * ffn_pre_gelu**3)
            )
        )
        + ffn_pre_gelu / 2
        * (
            1 - np.tanh(
                np.sqrt(2 / np.pi) *
                (ffn_pre_gelu + 0.044715 * ffn_pre_gelu**3)
            )**2
        )
        * np.sqrt(2 / np.pi)
        * (1 + 0.134145 * ffn_pre_gelu**2)
    ) * gelu_out_grad

    # Expansion Matrix
    ffn_main_grad = pre_gelu_grad @ expand_matrix.swapaxes(-2, -1)
    expand_matrix_grad = ffn_input.swapaxes(-2, -1) @ pre_gelu_grad

    # Residual
    ffn_input_grad = ffn_out_grad + ffn_main_grad

    # Attention
    # Layer Norm
    attn_ln_jacobian = layernorm_der(
        attn_std, attn_pre_norm, attn_norm, gamma_atn
    )
    attn_ln_input_grad = np.sum(
        attn_ln_jacobian * ffn_input_grad[:, :, None], axis=1
    )

    # Softmaxxed @ V
    attn_probs_grad = attn_ln_input_grad @ V.swapaxes(-2, -1)

    # V Gradient
    V_grad = attn_probs.swapaxes(-2, -1) @ attn_ln_input_grad

    # Softmax
    attn_scores_grad = attn_probs * (
        attn_probs_grad
        - np.sum(
            attn_probs * attn_probs_grad,
            axis=1,
            keepdims=True
        )
    )

    scores_grad = attn_scores_grad / np.sqrt(K.shape[-1])

    # Scores: Q and K
    Q_grad = scores_grad @ K
    K_grad = scores_grad.swapaxes(-2, -1) @ Q

    # Gradients for training
    q_training = scores_grad @ K
    k_training = scores_grad.swapaxes(-2,-1) @ Q
    v_training = attn_probs.swapaxes(-2,-1) @ attn_ln_input_grad

    QM_grad = EM.swapaxes(-2,-1) @ q_training
    KM_grad = EM.swapaxes(-2,-1) @ k_training 
    VM_grad = EM.swapaxes(-2,-1) @ v_training

    # Gradients flowing back into EM
    em_q_grad = Q_grad @ QM.swapaxes(-2, -1)
    em_k_grad = K_grad @ KM.swapaxes(-2, -1)
    em_v_grad = V_grad @ VM.swapaxes(-2, -1)

    em_grad = (
        em_q_grad + em_k_grad + em_v_grad + attn_ln_input_grad
    )

    # Token Embeddings
    np.add.at(embed_matrix_grad, batch, em_grad)

    # Gamma & Beta (both layernorns)
    ln_gamma_grad = np.sum(ffn_out_grad * pre_gamma_ffn_ln, axis=0)
    ln_beta_grad = np.sum(ffn_out_grad, axis=0)
    
    attn_gamma_grad = np.sum(ffn_input_grad * pre_gamma_attn_ln, axis=0) 
    attn_beta_grad = np.sum(ffn_input_grad, axis=0)
    grads = [
        embed_matrix_grad,
        output_matrix_grad,
        QM_grad,
        KM_grad,
        VM_grad,
        expand_matrix_grad,
        shrink_matrix_grad,
        ln_gamma_grad,
        ln_beta_grad,
        attn_gamma_grad,
        attn_beta_grad
    ]
    return grads

def layernorm_der(deviation, before_norm, normalized, gamma):
    layernorm_grad = (np.eye(standard_weights) - (1 / standard_weights)) / deviation.reshape(deviation.shape + (1,)) 
    center_values = before_norm - (np.mean(before_norm, axis=1, keepdims=True))
    center_values_copy = center_values.copy()
    centered_values_reshape = center_values.reshape((normalized.shape + (1,)))
    normalized_values_reshape = center_values_copy.reshape((((normalized.shape[0],) + (1,) + (normalized.shape[1],))))
    centered_outer_product = centered_values_reshape @ normalized_values_reshape
    variance_correction = centered_outer_product / (deviation.reshape(deviation.shape + (1,))**3 * standard_weights)
    layernorm_grad = layernorm_grad - variance_correction
    return layernorm_grad * gamma.reshape((1,) + (standard_weights,) + (1,))
vocabulary = data()
numberify()
embed_matrix = np.random.randn(number_tokens, standard_weights) #Amount of Tokens, Weights in the LLM
output_matrix = np.random.randn(standard_weights, number_tokens)
def training():
    global done
    useful_prompt, deviation_atn, pre_norm_atn, values, residual_input, softmaxxed, Q, K, EM, attn_pre_gamma, attn_pre_beta = attention()
    output, normalized_matrix, before_norm, deviation, gelu_output, expanded_matrix, shrink_matrix, ffn_matrix, total_beta, total_gamma = FFN(useful_prompt) 
    x,y,z = cross_entropy(output)
    backprop(y,z,output, normalized_matrix, before_norm, deviation, gelu_output, expanded_matrix, shrink_matrix, ffn_matrix, deviation_atn, pre_norm_atn, useful_prompt, values, residual_input, softmaxxed, Q, K, EM, total_gamma, total_beta, attn_pre_gamma, attn_pre_beta)
    update()
    return x

def generation():
    text = []
    for x in range(10):
        Attention = attention()[0]
        ffn = FFN(Attention)[0]
        logits = ffn @ output_matrix
        logits_MAXED = softmax(logits[-1:])
        text.append(int(np.argmax(logits_MAXED)))

    decoded_bytes = bytearray()
    for number in text:
        if number <= 255:
            decoded_bytes.append(number)
        else:
            decoded_bytes.extend(vocabulary[str(number)])

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    print(decoded_text)
    return decoded_text

parameters = [
    ("embed_matrix", embed_matrix),
    ("output_matrix", output_matrix),
    ("QM", QM),
    ("KM", KM),
    ("VM", VM),
    ("expand_matrix", expand_matrix),
    ("shrink_matrix", shrink_matrix),
    ("gamma_ffn", gamma_ffn),
    ("beta_ffn", beta_ffn),
    ("gamma_atn", gamma_atn),
    ("beta_atn", beta_atn),
]
for awewaefwaefwef in range(10):
    for n in range(100000):
        x = training()
        if done is True:
            start = 0
            print(x)
            print(n)
            break
    done = False
text = []
for n in range(20):
    x = generation()
    text.append(x)
print(text, "\n\n\n")