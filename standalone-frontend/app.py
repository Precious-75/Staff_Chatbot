from flask import Flask,render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"answer": "Please enter a message."})
    response, _confidence = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":  
    app.run(debug=True) 
