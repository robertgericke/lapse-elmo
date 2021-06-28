import torch
from typing import Dict, List


def load_vocab(filename: str) -> Dict[str, int]:
    vocab = {'<UNK>': 1}
    next_id = 2
    with open(filename) as file:
        for line in file:
            word = line.strip()
            if word not in vocab:
                vocab[word] = next_id
                next_id += 1
    return vocab


def batch_to_word_ids(batch: List[List[str]], vocab: Dict[str, int], max_sequence_length:int=200, dtype=torch.long, device=None) -> torch.Tensor:
    max_len = max(min(len(sentence), max_sequence_length) for sentence in batch)

    rows = []
    for sentence in batch:
        row = [vocab.get(word, 1) for word in sentence]
        if len(row) > max_sequence_length:
            row = row[:max_sequence_length]
        row.extend([0] * (max_len - len(row)))
        rows.append(row)

    return torch.tensor(rows, dtype=dtype, device=device)
