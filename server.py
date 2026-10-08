```python
import os
import base64
import requests
import torch

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import MiniAI
from tokenizer import Tokenizer


# ============================================================
# SERVER
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# EINSTELLUNGEN
# ============================================================

MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"

DATA_FOLDER = "DATEN"

GITHUB_API = "https://api.github.com"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_BRANCH = os.environ.get(
    "GITHUB_BRANCH",
    "main"
)


# ============================================================
# GITHUB
# ============================================================

def github_headers():

    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN fehlt."
        )

    return {
        "Accept":
            "application/vnd.github+json",

        "Authorization":
            f"Bearer {GITHUB_TOKEN}",

        "X-GitHub-Api-Version":
            "2026-03-10"
    }


def github_file_sha(path):

    url = (
        f"{GITHUB_API}/repos/"
        f"{GITHUB_REPO}/contents/"
        f"{path}"
    )

    response = requests.get(
        url,
        headers=github_headers(),
        params={
            "ref": GITHUB_BRANCH
        },
        timeout=30
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()

    return response.json()["sha"]


def upload_to_github(
    local_file,
    github_path,
    commit_message
):

    if not GITHUB_REPO:
        raise RuntimeError(
            "GITHUB_REPO fehlt."
        )


    if not os.path.exists(local_file):
        raise FileNotFoundError(
            local_file
        )


    print(
        f"[GITHUB] Lade {local_file} hoch..."
    )


    with open(
        local_file,
        "rb"
    ) as f:

        content = base64.b64encode(
            f.read()
        ).decode("ascii")


    sha = github_file_sha(
        github_path
    )


    url = (
        f"{GITHUB_API}/repos/"
        f"{GITHUB_REPO}/contents/"
        f"{github_path}"
    )


    payload = {

        "message":
            commit_message,

        "content":
            content,

        "branch":
            GITHUB_BRANCH
    }


    if sha:
        payload["sha"] = sha


    response = requests.put(
        url,
        headers=github_headers(),
        json=payload,
        timeout=120
    )


    if not response.ok:

        print(
            "[GITHUB ERROR]",
            response.text
        )

        response.raise_for_status()


    result = response.json()

    print(
        f"[GITHUB] {github_path} gespeichert."
    )

    return result


# ============================================================
# TOKENIZER + MODELL
# ============================================================

tokenizer = None
model = None


def load_model():

    global tokenizer
    global model


    if not os.path.exists(
        MODEL_FILE
    ):

        print(
            "[MODEL] model.pt nicht vorhanden."
        )

        return False


    if not os.path.exists(
        TOKENIZER_FILE
    ):

        print(
            "[MODEL] tokenizer.json nicht vorhanden."
        )

        return False


    print(
        "[MODEL] Lade Tokenizer..."
    )


    tokenizer = Tokenizer()

    tokenizer.load(
        TOKENIZER_FILE
    )


    print(
        "[MODEL] Lade model.pt..."
    )


    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )


    model = MiniAI(

        vocab_size=
            checkpoint["vocab_size"],

        embedding_size=
            checkpoint["embedding_size"],

        hidden_size=
            checkpoint["hidden_size"]
    )


    model.load_state_dict(
        checkpoint["model_state"]
    )


    model.eval()


    print(
        "[MODEL] Modell geladen."
    )


    return True


# ============================================================
# TRAINING
# ============================================================

def train_model():

    print()
    print(
        "========================================"
    )
    print(
        "             TRAINING"
    )
    print(
        "========================================"
    )


    if not os.path.exists(
        DATA_FOLDER
    ):

        raise RuntimeError(
            "DATEN-Ordner fehlt."
        )


    all_text = []


    files = sorted(
        os.listdir(DATA_FOLDER)
    )


    for filename in files:

        if not filename.lower().endswith(
            ".txt"
        ):
            continue


        path = os.path.join(
            DATA_FOLDER,
            filename
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()


        if content.strip():

            all_text.append(
                content
            )


        print(
            f"[DATEN] {filename}: "
            f"{len(content)} Zeichen"
        )


    if not all_text:

        raise RuntimeError(
            "Keine TXT-Dateien gefunden."
        )


    text = "\n\n".join(
        all_text
    )


    print(
        f"[DATEN] Gesamt: "
        f"{len(text)} Zeichen"
    )


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    new_tokenizer = Tokenizer()

    new_tokenizer.build(
        text
    )


    new_tokenizer.save(
        TOKENIZER_FILE
    )


    data = torch.tensor(
        new_tokenizer.encode(text),
        dtype=torch.long
    )


    vocab_size = len(
        new_tokenizer.stoi
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    BLOCK_SIZE = 16

    EMBEDDING_SIZE = 128

    HIDDEN_SIZE = 256

    EPOCHS = int(
        os.environ.get(
            "TRAIN_EPOCHS",
            "300"
        )
    )

    LEARNING_RATE = 0.003


    if len(data) <= BLOCK_SIZE:

        raise RuntimeError(
            "Zu wenig Trainingsdaten."
        )


    inputs = []
    targets = []


    for i in range(
        len(data) - BLOCK_SIZE
    ):

        inputs.append(
            data[
                i:i + BLOCK_SIZE
            ]
        )

        targets.append(
            data[
                i + 1:i + BLOCK_SIZE + 1
            ]
        )


    X = torch.stack(
        inputs
    )

    Y = torch.stack(
        targets
    )


    print(
        f"[MODEL] Vokabular: "
        f"{vocab_size}"
    )

    print(
        f"[MODEL] Sequenzen: "
        f"{len(X)}"
    )


    new_model = MiniAI(

        vocab_size=vocab_size,

        embedding_size=
            EMBEDDING_SIZE,

        hidden_size=
            HIDDEN_SIZE
    )


    optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=LEARNING_RATE
    )


    loss_function = (
        torch.nn.CrossEntropyLoss()
    )


    for epoch in range(EPOCHS):

        new_model.train()

        optimizer.zero_grad()


        output, _ = new_model(X)


        loss = loss_function(

            output.reshape(
                -1,
                vocab_size
            ),

            Y.reshape(-1)
        )


        loss.backward()

        optimizer.step()


        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
        ):

            print(
                f"[TRAIN] "
                f"{epoch + 1}/{EPOCHS} "
                f"Loss: "
                f"{loss.item():.6f}"
            )


    # --------------------------------------------------------
    # MODEL SPEICHERN
    # --------------------------------------------------------

    torch.save(

        {
            "model_state":
                new_model.state_dict(),

            "vocab_size":
                vocab_size,

            "embedding_size":
                EMBEDDING_SIZE,

            "hidden_size":
                HIDDEN_SIZE,

            "block_size":
                BLOCK_SIZE,

            "epochs":
                EPOCHS,

            "training_tokens":
                len(data)

        },

        MODEL_FILE
    )


    # --------------------------------------------------------
    # GLOBAL AKTUALISIEREN
    # --------------------------------------------------------

    tokenizer = new_tokenizer

    model = new_model

    model.eval()


    print(
        "[TRAIN] Training fertig."
    )


    # --------------------------------------------------------
    # GITHUB
    # --------------------------------------------------------

    upload_to_github(

        TOKENIZER_FILE,

        TOKENIZER_FILE,

        "Update tokenizer"
    )


    upload_to_github(

        MODEL_FILE,

        MODEL_FILE,

        "Update trained model"
    )


    print(
        "[GITHUB] Modell und Tokenizer "
        "wurden gespeichert."
    )


    return True


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
# STATUS
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def status():

    return jsonify({

        "online": True,

        "model_loaded":
            model is not None,

        "github":
            bool(GITHUB_TOKEN and GITHUB_REPO)

    })


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    global model


    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "error":
                    "Kein JSON erhalten."
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
                "error":
                    "message muss Text sein."
            }), 400


        message = message.strip()


        if not message:

            return jsonify({
                "error":
                    "Nachricht ist leer."
            }), 400


        if model is None:

            return jsonify({

                "error":
                    "Kein trainiertes Modell vorhanden."

            }), 503


        ids = tokenizer.encode(
            message
        )


        if not ids:

            return jsonify({

                "response":
                    "Ich konnte deine Nachricht nicht verarbeiten."

            })


        x = torch.tensor(
            [ids],
            dtype=torch.long
        )


        with torch.no_grad():

            output, hidden = model(
                x
            )


            for _ in range(50):

                last = output[:, -1, :]


                probabilities = torch.softmax(
                    last,
                    dim=-1
                )


                next_token = torch.multinomial(
                    probabilities,
                    1
                )


                x = torch.cat(
                    [x, next_token],
                    dim=1
                )


                output, hidden = model(
                    next_token,
                    hidden
                )


        answer = tokenizer.decode(
            x[0].tolist()
        )


        if answer.lower().startswith(
            message.lower()
        ):

            answer = answer[
                len(message):
            ].strip()


        return jsonify({

            "success": True,

            "response":
                answer

        })


    except Exception as error:

        print(
            "[CHAT ERROR]",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# TRAINING API
# ============================================================

@app.route(
    "/api/train",
    methods=["POST"]
)
def train_api():

    try:

        train_model()


        return jsonify({

            "success": True,

            "message":
                "Training abgeschlossen.",

            "model":
                MODEL_FILE,

            "tokenizer":
                TOKENIZER_FILE

        })


    except Exception as error:

        print(
            "[TRAIN ERROR]",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )
    print(
        "             MEINE AI"
    )
    print(
        "========================================"
    )


    # Falls bereits ein Modell vorhanden ist:
    # direkt laden.

    if not load_model():

        print(
            "[START] Noch kein Modell vorhanden."
        )

        print(
            "[START] Training kann über "
            "/api/train gestartet werden."
        )


    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )


    app.run(
        host="0.0.0.0",
        port=port
    )
```
