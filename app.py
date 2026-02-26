from flask import Flask, render_template, request, jsonify
from rag import ask, deep_think_ask

app = Flask(__name__)

MODELS = [
    {"id": "gpt-4o-mini",  "label": "GPT-4o mini",  "desc": "Fast & cheap"},
    {"id": "gpt-4o",       "label": "GPT-4o",        "desc": "Best quality"},
    {"id": "o3-mini",      "label": "o3-mini",       "desc": "Deep reasoning"},
]

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    question   = data.get("message", "").strip()
    model      = data.get("model", "gpt-4o-mini")
    top_k      = max(1, min(int(data.get("top_k", 5)), 20))
    deep_think = bool(data.get("deep_think", False))
    iterations = max(1, min(int(data.get("iterations", 2)), 5))

    if not question:
        return jsonify({"error": "Empty message"}), 400

    # Validate model
    valid_ids = [m["id"] for m in MODELS]
    if model not in valid_ids:
        model = "gpt-4o-mini"

    try:
        if deep_think:
            result = deep_think_ask(question, model=model, top_k=top_k, iterations=iterations)
        else:
            result = ask(question, model=model, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
