import json
import numpy as np
from typing import List
from torch_geometric.data import Data
import torch

def load_and_tokenize(dataset_path: str, tokenizer: JAIDETokenizer) -> List[Data]:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    processed_data = []
    for item in dataset:
        text = item['text']
        tokens = tokenizer.encode(text)
        graph_data = tokens_to_graph(tokens)
        processed_data.append(graph_data)
    return processed_data

def tokens_to_graph(tokens: List[int]) -> Data:
    edge_index = []
    num_tokens = len(tokens)
    synastry_matrix = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            if (tokens[i] + tokens[j]) % 3 == 0:
                synastry_matrix[i, j] = 1
                synastry_matrix[j, i] = 1

    for i in range(num_tokens - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            if synastry_matrix[i, j] != 0:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(-1)
    return Data(x=x, edge_index=edge_index)
