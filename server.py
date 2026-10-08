from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import os

from model import MiniAI
from tokenizer import Tokenizer


# ============================================================
# SERVER
# ============================================================

app = Flask(__name__, static_folder=".")
CORS(app)


# ============================================================
# DATEIEN
# ============================================================

MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"


# ============================================================
# KI LADEN
# ============================================================

print("[START] Meine AI wird geladen...")


if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "model.pt wurde nicht gefunden."
    )


if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(
        "tokenizer.json wurde nicht gefunden."
    )


tokenizer = Tokenizer()
tokenizer.load(TOKENIZER_FILE)


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


print("[START] Meine AI ist bereit.")
print(
    "[MODEL] Vokabular:",
    checkpoint["vocab_size"]
)


# ============================================================
# TEXT GENERIEREN
# ============================================================

def generate(prompt, amount=50):

    ids = tokenizer.encode(prompt)

    if not ids:
        return "Ich konnte deine Nachricht nicht verstehen."


    x = torch.tensor(
        [ids],
        dtype=torch.long
    )


    with torch.no_grad():

        output, hidden = model(x)


        for _ in range(amount):

            last = output[:, -1, :]

            probabilities = torch.softmax(
                last,
                dim=-1
            )


            # Zufällige Auswahl aus den wahrscheinlichsten
            # Tokens

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


    result = tokenizer.decode(
        x[0].tolist()
    )


    # Prompt nicht doppelt zurückgeben

    if result.lower().startswith(
        prompt.lower()
    ):

        result = result[len(prompt):].strip()


    return result.strip()


# ============================================================
# STARTSEITE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "error": "Kein JSON erhalten."
            }), 400


        message = data.get(
            "message",
            ""
        )


        if not isinstance(
            message,
            str
        ):

            return jsonify({
                "error": "message muss Text sein."
            }), 400


        message = message.strip()


        if not message:

            return jsonify({
                "error": "Nachricht ist leer."
            }), 400


        print(
            "[CHAT]",
            message
        )


        answer = generate(
            message,
            amount=50
        )


        print(
            "[AI]",
            answer
        )


        return jsonify({

            "success": True,

            "message": message,

            "response": answer

        })


    except Exception as error:

        print(
            "[ERROR]",
            error
        )


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():

    return jsonify({

        "online": True,

        "ai": "Meine AI",

        "model_loaded": True

    })


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    print()
    print(
        "[SERVER] Port:",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
