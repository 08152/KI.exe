import torch

from tokenizer import Tokenizer
from model import MiniAI


MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"


# =========================
# Tokenizer laden
# =========================

tokenizer = Tokenizer()
tokenizer.load(TOKENIZER_FILE)


# =========================
# Modell laden
# =========================

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

model = MiniAI(
    vocab_size=checkpoint["vocab_size"],
    embedding_size=checkpoint["embedding_size"],
    hidden_size=checkpoint["hidden_size"]
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


# =========================
# Text erzeugen
# =========================

def generate(prompt, amount=30):

    ids = tokenizer.encode(prompt)

    if not ids:
        return ""

    x = torch.tensor(
        [ids],
        dtype=torch.long
    )

    hidden = None

    with torch.no_grad():

        output, hidden = model(
            x,
            hidden
        )

        for _ in range(amount):

            last = output[:, -1, :]

            probabilities = torch.softmax(
                last,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            x = torch.cat(
                [x, next_token],
                dim=1
            )

            output, hidden = model(
                next_token,
                hidden
            )

    return tokenizer.decode(
        x[0].tolist()
    )


# =========================
# Chat
# =========================

print("Meine AI ist bereit.")
print("Schreibe 'exit' zum Beenden.")
print()

while True:

    user = input("Du: ")

    if user.lower() == "exit":
        break

    answer = generate(
        user,
        amount=30
    )

    print("KI:", answer)
    print()
