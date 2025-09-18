import json
from typing import List
import torch
from torch import Tensor

class JAIDETokenizer:
    def __init__(self, vocab_path: str = "models/vocab.json"):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.max_seq_len = 16000000
        self.pad_token_id = self.vocab.get("<PAD>", 0)
        self.bos_token_id = self.vocab.get("<BOS>", 1)
        self.eos_token_id = self.vocab.get("<EOS>", 2)
        self.unk_token_id = self.vocab.get("<UNK>", 3)

    def encode(self, text: str) -> List[int]:
        tokens = [self.bos_token_id]
        for char in text:
            tokens.append(self.vocab.get(char, self.unk_token_id))
        tokens.append(self.eos_token_id)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens += [self.pad_token_id] * (self.max_seq_len - len(tokens))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            chars.append(self.inverse_vocab.get(tid, "<UNK>"))
        return "".join(chars)

    def batch_encode(self, texts: List[str]) -> Tensor:
        return torch.tensor([self.encode(text) for text in texts], dtype=torch.long)

    def batch_decode(self, token_ids: Tensor) -> List[str]:
        return [self.decode(ids.tolist()) for ids in token_ids]
