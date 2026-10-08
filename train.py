import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import Tokenizer
from model import MiniAI


# =========================
# Einstellungen
# =========================

DATA_FILE = "data.txt"
MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"

EPOCHS = 300
BLOCK_SIZE = 16
LEARNING_RATE = 0.003


# =========================
# Daten laden
# =========================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()


# =========================
# Tokenizer erstellen
# =========================

tokenizer = Tokenizer()
tokenizer.build(text)

tokenizer.save(TOKENIZER_FILE)

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

vocab_size = len(tokenizer.stoi)

print("Vokabular:", vocab_size)
print("Tokens:", len(data))


# =========================
# Trainingsdaten erzeugen
# =========================

inputs = []
targets = []

for i in range(len(data) - BLOCK_SIZE):
    x = data[i:i + BLOCK_SIZE]
    y = data[i + 1:i + BLOCK_SIZE + 1]

    inputs.append(x)
    targets.append(y)


X = torch.stack(inputs)
Y = torch.stack(targets)


# =========================
# Modell
# =========================

model = MiniAI(
    vocab_size=vocab_size,
    embedding_size=128,
    hidden_size=256
)


optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()


# =========================
# Training
# =========================

print()
print("TRAINING STARTET")
print("================")

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    output, _ = model(X)

    loss = loss_function(
        output.reshape(-1, vocab_size),
        Y.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d}/{EPOCHS} "
            f"Loss: {loss.item():.4f}"
        )


# =========================
# Modell speichern
# =========================

torch.save(
    {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_size": 128,
        "hidden_size": 256
    },
    MODEL_FILE
)

print()
print("================")
print("TRAINING FERTIG")
print("Gespeichert als:", MODEL_FILE)
print("Tokenizer:", TOKENIZER_FILE)
