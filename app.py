from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import predict_churn

app = Flask(__name__)
CORS(app)

@app.route("/dados", methods=["GET"])
def get_dados():
    # Retorna os dados simulados caso a página precise carregar algo no início
    dados = predict_churn(None, None) 
    return jsonify(dados)

@app.route("/upload", methods=["POST"])
def upload_file():
    # Valida se os dois arquivos obrigatórios foram enviados no FormData
    if 'clientes' not in request.files or 'catraca' not in request.files:
        return jsonify({"error": "É necessário enviar ambos os arquivos: 'clientes' e 'catraca'."}), 400
    
    file_clientes = request.files['clientes']
    file_catraca = request.files['catraca']
    
    if file_clientes.filename == '' or file_catraca.filename == '':
        return jsonify({"error": "Um ou ambos os arquivos selecionados estão vazios."}), 400

    # Passa as duas planilhas para o pipeline realizar o tratamento e predição real
    resultado = predict_churn(file_clientes, file_catraca)
    
    # Se o retorno for um dicionário contendo erro, muda o status HTTP
    if isinstance(resultado, dict) and "error" in resultado:
        return jsonify(resultado), 400
        
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True, port=5000)