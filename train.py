import os
import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import Tokenizer
from model import MiniAI


# ============================================================
# EINSTELLUNGEN
# ============================================================

DATA_FOLDER = "DATEN"

MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"

EPOCHS = 300
BLOCK_SIZE = 16
LEARNING_RATE = 0.003


# ============================================================
# DATEN AUS DATEN/ LADEN
# ============================================================

print()
print("========================================")
print("          MEINE AI - TRAINING")
print("========================================")
print()

if not os.path.exists(DATA_FOLDER):
    print("[FEHLER] Der Ordner DATEN existiert nicht.")
    print()
    print("Erstelle:")
    print("DATEN/")
    raise SystemExit


all_text = []

files = sorted(os.listdir(DATA_FOLDER))

txt_files = [
    filename
    for filename in files
    if filename.lower().endswith(".txt")
]


if not txt_files:
    print("[FEHLER] Keine TXT-Dateien in DATEN gefunden.")
    raise SystemExit


print("[DATEN] Dateien gefunden:")
print()


for filename in txt_files:

    path = os.path.join(
        DATA_FOLDER,
        filename
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        if content.strip():

            all_text.append(content)

            print(
                f"[OK] {filename} "
                f"({len(content):,} Zeichen)"
            )

        else:

            print(
                f"[WARNUNG] {filename} ist leer."
            )

    except Exception as error:

        print(
            f"[FEHLER] {filename}: {error}"
        )


# ============================================================
# GESAMTEN TEXT ZUSAMMENFÜGEN
# ============================================================

if not all_text:
    print()
    print("[FEHLER] Es wurden keine Daten geladen.")
    raise SystemExit


text = "\n\n".join(all_text)


print()
print("----------------------------------------")
print(
    f"[DATEN] Insgesamt: {len(text):,} Zeichen"
)
print(
    f"[DATEN] TXT-Dateien: {len(all_text)}"
)
print("----------------------------------------")
print()


# ============================================================
# TOKENIZER ERSTELLEN
# ============================================================

print("[TOKENIZER] Erstelle Vokabular...")

tokenizer = Tokenizer()

tokenizer.build(text)

tokenizer.save(TOKENIZER_FILE)

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

vocab_size = len(tokenizer.stoi)

print(
    f"[TOKENIZER] Vokabular: {vocab_size}"
)

print(
    f"[TOKENIZER] Tokens: {len(data):,}"
)

print()


# ============================================================
# TRAININGSDATEN ERSTELLEN
# ============================================================

if len(data) <= BLOCK_SIZE:
    print(
        "[FEHLER] Zu wenig Trainingsdaten."
    )

    print(
        f"Benötigt werden mehr als "
        f"{BLOCK_SIZE} Tokens."
    )

    raise SystemExit


print("[DATEN] Erstelle Trainingssequenzen...")


inputs = []
targets = []


for i in range(
    len(data) - BLOCK_SIZE
):

    x = data[
        i:i + BLOCK_SIZE
    ]

    y = data[
        i + 1:i + BLOCK_SIZE + 1
    ]

    inputs.append(x)
    targets.append(y)


X = torch.stack(inputs)

Y = torch.stack(targets)


print(
    f"[DATEN] Trainingssequenzen: {len(X):,}"
)

print()


# ============================================================
# MODELL ERSTELLEN
# ============================================================

print("[MODEL] Erstelle neuronales Netzwerk...")


EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256


model = MiniAI(
    vocab_size=vocab_size,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE
)


# Anzahl Parameter anzeigen

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    f"[MODEL] Parameter: {parameter_count:,}"
)

print()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


loss_function = nn.CrossEntropyLoss()


# ============================================================
# TRAINING
# ============================================================

print("========================================")
print("             TRAINING START")
print("========================================")
print()


for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()


    # Vorhersage

    output, _ = model(X)


    # Loss berechnen

    loss = loss_function(
        output.reshape(
            -1,
            vocab_size
        ),
        Y.reshape(-1)
    )


    # Backpropagation

    loss.backward()


    # Gewichte aktualisieren

    optimizer.step()


    # Fortschritt anzeigen

    if (
        epoch == 0
        or
        (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:4d}/{EPOCHS} "
            f"| Loss: "
            f"{loss.item():.6f}"
        )


# ============================================================
# MODELL SPEICHERN
# ============================================================

print()
print("========================================")
print("          SPEICHERE MODELL")
print("========================================")
print()


torch.save(
    {
        "model_state": model.state_dict(),

        "vocab_size": vocab_size,

        "embedding_size": EMBEDDING_SIZE,

        "hidden_size": HIDDEN_SIZE,

        "block_size": BLOCK_SIZE,

        "epochs": EPOCHS,

        "training_tokens": len(data),

        "training_files": txt_files

    },
    MODEL_FILE
)


# ============================================================
# ABSCHLUSS
# ============================================================

print(
    f"[OK] Modell gespeichert: {MODEL_FILE}"
)

print(
    f"[OK] Tokenizer gespeichert: "
    f"{TOKENIZER_FILE}"
)

print()
print("========================================")
print("          TRAINING FERTIG!")
print("========================================")
print()
print("Starte deine KI mit:")
print()
print("py chat.py")
print()
