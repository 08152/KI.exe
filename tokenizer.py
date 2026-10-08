import re
import json
import os


class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def build(self, text):
        # Wörter und Satzzeichen getrennt erkennen
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

        vocabulary = sorted(set(tokens))

        # Sonderzeichen
        vocabulary = ["<PAD>", "<UNK>"] + vocabulary

        self.stoi = {token: i for i, token in enumerate(vocabulary)}
        self.itos = {i: token for token, i in self.stoi.items()}

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]

    def decode(self, ids):
        tokens = [self.itos.get(int(i), "<UNK>") for i in ids]

        result = ""

        for token in tokens:
            if token in ".,!?;:":
                result += token
            else:
                if result:
                    result += " "
                result += token

        return result

    def save(self, path):
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()}
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
