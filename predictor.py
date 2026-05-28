import time
import random

def predict_churn(data):
    """
    Simulates the prediction logic from 'another code'.
    In a real scenario, this could call an ML model or a separate service.
    """
    print("Iniciando predição...")
    # Simula o tempo de espera da predição
    time.sleep(2) 
    
    # Gera dados de resposta simulados baseados no cenário de academia
    alunos = [
        {"nome": "Ana Silva", "probabilidade": random.uniform(0.7, 0.95)},
        {"nome": "Claudio Ferreira", "probabilidade": random.uniform(0.1, 0.3)},
        {"nome": "Fernando Costa", "probabilidade": random.uniform(0.8, 0.99)},
        {"nome": "Renan Santos", "probabilidade": random.uniform(0.01, 0.1)},
        {"nome": "Juliana Lima", "probabilidade": random.uniform(0.4, 0.7)},
        {"nome": "Marcos Souza", "probabilidade": random.uniform(0.3, 0.5)},
        {"nome": "Beatriz Oliveira", "probabilidade": random.uniform(0.05, 0.2)},
        {"nome": "Ricardo Gomes", "probabilidade": random.uniform(0.6, 0.85)}
    ]
    
    print("Predição concluída.")
    return alunos
