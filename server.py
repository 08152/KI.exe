import os
import json
import base64
import requests
import torch
import torch.nn as nn

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import MiniAI
from tokenizer import SimpleTokenizer


app = Flask(__name__)
CORS(app)

MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"
DATA_FOLDER = "DATEN"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "300"))

model = None
tokenizer = None


# --------------------------------------------------
# GITHUB
# --------------------------------------------------

def github_headers():
    if not GITHUB_TOKEN:
        return None

    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10"
    }


def github_file_sha(path):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    response = requests.get(
        url,
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=30
    )

    if response.status_code == 200:
        return response.json().get("sha")

    if response.status_code == 404:
        return None

    print("GitHub GET Fehler:", response.status_code)
    print(response.text)

    return None


def upload_to_github(local_file, github_path, commit_message):
    if not GITHUB_TOKEN:
        print("Kein GITHUB_TOKEN vorhanden.")
        return False

    if not GITHUB_REPO:
        print("Keine GITHUB_REPO Variable vorhanden.")
        return False

    if not os.path.exists(local_file):
        print("Datei nicht gefunden:", local_file)
        return False

    with open(local_file, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{github_path}"

    data = {
        "message": commit_message,
        "content": content,
        "branch": GITHUB_BRANCH
    }

    sha = github_file_sha(github_path)

    if sha:
        data["sha"] = sha

    response = requests.put(
        url,
        headers=github_headers(),
        json=data,
        timeout=120
    )

    if response.status_code in (200, 201):
        print("GitHub Upload erfolgreich:", github_path)
        return True

    print("GitHub Upload fehlgeschlagen:", response.status_code)
    print(response.text)

    return False


# --------------------------------------------------
# MODELL LADEN
# --------------------------------------------------

def load_model():
    global model
    global tokenizer

    if not os.path.exists(MODEL_FILE):
        print("[MODEL] model.pt nicht gefunden.")
        return False

    if not os.path.exists(TOKENIZER_FILE):
        print("[MODEL] tokenizer.json nicht gefunden.")
        return False

    try:
        tokenizer = SimpleTokenizer.load(TOKENIZER_FILE)

        model = MiniAI(
            vocab_size=len(tokenizer.vocab)
        )

        checkpoint = torch.load(
            MODEL_FILE,
            map_location="cpu"
        )

        model.load_state_dict(checkpoint)

        model.eval()

        print("[MODEL] Modell erfolgreich geladen.")
        print("[MODEL] Vokabular:", len(tokenizer.vocab))

        return True

    except Exception as e:
        print("[MODEL] Fehler beim Laden:", e)

        model = None
        tokenizer = None

        return False


# --------------------------------------------------
# TEXT GENERIEREN
# --------------------------------------------------

def generate_text(prompt, max_tokens=40):
    if model is None or tokenizer is None:
        return "Das Modell ist noch nicht trainiert."

    try:
        tokens = tokenizer.encode(prompt)

        if not tokens:
            tokens = [0]

        input_ids = torch.tensor(
            [tokens],
            dtype=torch.long
        )

        hidden = None

        generated = tokens[:]

        with torch.no_grad():

            for _ in range(max_tokens):

                output, hidden = model(
                    input_ids,
                    hidden
                )

                logits = output[:, -1, :]

                # Temperatur
                temperature = 0.8
                logits = logits / temperature

                probabilities = torch.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

                token_id = next_token.item()

                generated.append(token_id)

                input_ids = next_token

        text = tokenizer.decode(generated)

        # Prompt möglichst nicht doppelt anzeigen
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text.strip()

    except Exception as e:
        print("[GENERATE ERROR]", e)
        return "Beim Generieren ist ein Fehler aufgetreten."


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_model():
    global model
    global tokenizer

    print("[TRAIN] Training gestartet.")

    if not os.path.exists(DATA_FOLDER):
        print("[TRAIN] DATEN-Ordner nicht gefunden.")
        return False

    text_files = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".txt"):
            text_files.append(
                os.path.join(DATA_FOLDER, filename)
            )

    if not text_files:
        print("[TRAIN] Keine .txt-Dateien gefunden.")
        return False

    print("[TRAIN] Gefundene Dateien:", len(text_files))

    all_text = ""

    for filename in text_files:

        try:
            with open(
                filename,
                "r",
                encoding="utf-8"
            ) as f:

                content = f.read()

            all_text += "\n" + content

            print(
                "[TRAIN] Geladen:",
                filename,
                len(content),
                "Zeichen"
            )

        except Exception as e:
            print(
                "[TRAIN] Fehler bei",
                filename,
                e
            )

    if not all_text.strip():
        print("[TRAIN] Trainingsdaten sind leer.")
        return False

    # Tokenizer neu erstellen
    tokenizer = SimpleTokenizer()

    tokenizer.build(all_text)

    encoded = tokenizer.encode(all_text)

    if len(encoded) < 3:
        print("[TRAIN] Zu wenig Trainingsdaten.")
        return False

    print(
        "[TRAIN] Tokens:",
        len(encoded)
    )

    print(
        "[TRAIN] Vokabular:",
        len(tokenizer.vocab)
    )

    model = MiniAI(
        vocab_size=len(tokenizer.vocab)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    block_size = 16

    for epoch in range(TRAIN_EPOCHS):

        total_loss = 0.0
        steps = 0

        for i in range(
            0,
            len(encoded) - block_size - 1,
            block_size
        ):

            x = torch.tensor(
                encoded[i:i + block_size],
                dtype=torch.long
            ).unsqueeze(0)

            y = torch.tensor(
                encoded[i + 1:i + block_size + 1],
                dtype=torch.long
            ).unsqueeze(0)

            optimizer.zero_grad()

            output, _ = model(x)

            loss = loss_function(
                output.reshape(-1, output.shape[-1]),
                y.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:

            average_loss = (
                total_loss / steps
                if steps > 0
                else 0
            )

            print(
                f"[TRAIN] "
                f"Epoch {epoch + 1}/{TRAIN_EPOCHS} "
                f"Loss: {average_loss:.4f}"
            )

    model.eval()

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    tokenizer.save(
        TOKENIZER_FILE
    )

    print("[TRAIN] model.pt gespeichert.")
    print("[TRAIN] tokenizer.json gespeichert.")

    # GitHub
    upload_to_github(
        MODEL_FILE,
        MODEL_FILE,
        "Update model.pt"
    )

    upload_to_github(
        TOKENIZER_FILE,
        TOKENIZER_FILE,
        "Update tokenizer.json"
    )

    print("[TRAIN] Training abgeschlossen.")

    return True


# --------------------------------------------------
# WEB
# --------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(
        ".",
        "index.html"
    )


@app.route("/api/status")
def status():

    return jsonify({
        "online": True,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "github_configured": bool(
            GITHUB_TOKEN and GITHUB_REPO
        )
    })


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    message = str(
        data.get("message", "")
    ).strip()

    if not message:
        return jsonify({
            "error": "Keine Nachricht."
        }), 400

    answer = generate_text(
        message,
        max_tokens=40
    )

    return jsonify({
        "response": answer
    })


@app.route("/api/train", methods=["POST"])
def train():

    success = train_model()

    if success:
        return jsonify({
            "success": True,
            "message": "Training erfolgreich."
        })

    return jsonify({
        "success": False,
        "message": "Training fehlgeschlagen."
    }), 500


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":

    print("[START] KI.exe Server startet...")

    load_model()

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    print(
        f"[START] Server läuft auf Port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
```
