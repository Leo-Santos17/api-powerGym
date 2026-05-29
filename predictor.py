import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------------
# INICIALIZAÇÃO: Carrega os modelos globais em memória
# -------------------------------------------------------------------------
MODEL_PATH = 'modelo_churn.h5'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

model = None
scaler = None
label_encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("--- MODELOS REAIS CARREGADOS COM SUCESSO ---")
else:
    print("--- AVISO: Arquivos do modelo não encontrados. Rodando modo simulação. ---")


def calcular_idade(nascimento):
    if pd.isnull(nascimento):
        return None
    hoje = datetime.now()
    return hoje.year - nascimento.year - ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))


def executar_predicao_real(file_clientes, file_catraca):
    """
    Executa o pipeline completo de tratamento estrito e predição ML.
    """
    # 1. Tratamento da Base de Clientes (CSV) com fallback de encoding
    colunas_clientes = ['Nome', 'Situação do contrato', 'Situação do cliente', 'Data de nascimento', 'Sexo']
    
    try:
        # Tenta ler no padrão moderno (UTF-8)
        df_clientes = pd.read_csv(file_clientes, usecols=colunas_clientes, encoding='utf-8')
    except UnicodeDecodeError:
        # Se der erro, volta para o início do arquivo e tenta com a codificação do Excel/Windows BR
        file_clientes.seek(0) # Reseta o ponteiro de leitura do arquivo enviado pela API
        df_clientes = pd.read_csv(file_clientes, usecols=colunas_clientes, encoding='iso-8859-1')
    
    # Limpeza estrita de nulos nas colunas fundamentais
    df_clientes = df_clientes.dropna(subset=['Nome', 'Data de nascimento', 'Sexo'])
    df_clientes['Nome'] = df_clientes['Nome'].astype(str).str.strip()
    
    # Conversão de data e cálculo de idade
    df_clientes['Data de nascimento'] = pd.to_datetime(df_clientes['Data de nascimento'], errors='coerce')
    df_clientes = df_clientes.dropna(subset=['Data de nascimento'])
    df_clientes['Idade'] = df_clientes['Data de nascimento'].apply(calcular_idade)
    
    # 2. Tratamento da Catraca (Mapeando 'Cliente' para 'Nome')
    colunas_catraca = ['Cliente', 'Contrato', 'Data']
    
    # Verifica a extensão para ler CSV ou Excel
    if hasattr(file_catraca, 'filename') and file_catraca.filename.endswith('.csv'):
        df_catraca = pd.read_csv(file_catraca, usecols=colunas_catraca)
    else:
        df_catraca = pd.read_excel(file_catraca, usecols=colunas_catraca)
        
    df_catraca = df_catraca.rename(columns={'Cliente': 'Nome'})
    
    # Limpeza estrita da Catraca
    df_catraca = df_catraca.dropna(subset=['Nome', 'Data'])
    df_catraca['Nome'] = df_catraca['Nome'].astype(str).str.strip()
    
    # Remove termos inválidos do sistema
    termos_invalidos = ['CANCELADO', 'diaria 7', 'nan', '']
    df_catraca = df_catraca[~df_catraca['Nome'].isin(termos_invalidos)]
    
    df_catraca['Data'] = pd.to_datetime(df_catraca['Data'], errors='coerce')
    df_catraca = df_catraca.dropna(subset=['Data'])
    
    # 3. Merge e Agrupamento (Feature Engineering)
    base_mista = df_catraca.merge(df_clientes[['Nome', 'Sexo', 'Idade']], on='Nome', how='inner')
    
    if base_mista.empty:
        return {"error": "Nenhum dado compatível encontrado após a limpeza das planilhas."}
        
    df_features = base_mista.groupby('Nome').agg(
        Total_Visitas=('Data', 'count'),
        Idade=('Idade', 'first'),
        Sexo=('Sexo', 'first')
    ).reset_index()
    
    # 4. Label Encoding do Sexo (Evitando falhas com classes novas)
    df_features = df_features[df_features['Sexo'].isin(label_encoder.classes_)]
    df_features['Sexo'] = label_encoder.transform(df_features['Sexo'].astype(str))
    
    # 5. Predição com o Modelo Keras
    X_raw = df_features[['Total_Visitas', 'Idade', 'Sexo']].values
    X_scaled = scaler.transform(X_raw)
    
    predicoes_prob = model.predict(X_scaled).flatten()
    lista_nomes = df_features['Nome'].tolist()
    
    # Formata o retorno exatamente como sua página web espera
    resultado = [
        {"nome": nome, "probabilidade": float(prob)}
        for nome, prob in zip(lista_nomes, predicoes_prob)
    ]
    
    return resultado


def predict_churn(file_clientes=None, file_catraca=None):
    """
    Função principal adaptada para manter compatibilidade com sua rota antiga (GET /dados)
    e processar dados reais no POST /upload.
    """
    # Se não enviar os arquivos (ex: rota GET /dados), mantém um Mock de segurança
    if file_clientes is None or file_catraca is None or model is None:
        return [
            {"nome": "Ana Silva (Simulado)", "probabilidade": 0.8146},
            {"nome": "Claudio Ferreira (Simulado)", "probabilidade": 0.1523},
            {"nome": "Fernando Costa (Simulado)", "probabilidade": 0.9241}
        ]
    
    return executar_predicao_real(file_clientes, file_catraca)