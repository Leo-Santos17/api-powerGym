from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import predict_churn
import os

app = Flask(__name__)
CORS(app)

@app.route("/dados", methods=["GET"])
def get_dados():
    # Retorna dados iniciais simulados
    dados = predict_churn(None) # Usa o predictor para gerar dados iniciais
    return jsonify(dados)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Aqui você salvaria o arquivo se necessário
    # file.save(os.path.join("uploads", file.filename))
    # Chama a predição e espera a resposta
    resultado = predict_churn(file)
    
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
