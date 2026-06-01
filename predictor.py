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
    # 1. Tratamento da Base de Clientes (Híbrido: Excel ou CSV)
    colunas_clientes = ['Nome', 'Situação do contrato', 'Situação do cliente', 'Data de nascimento', 'Sexo']

    if hasattr(file_clientes, 'filename') and (file_clientes.filename.endswith('.xlsx') or file_clientes.filename.endswith('.xls')):
        # Leitura de Excel: Lê completo e higieniza os títulos das colunas
        df_temp = pd.read_excel(file_clientes)
        df_temp.columns = df_temp.columns.astype(str).str.strip()
        df_clientes = df_temp[colunas_clientes]
    else:
        # Leitura de CSV: Aplica a proteção de encoding, separador (; ou ,) e espaços invisíveis
        try:
            df_temp = pd.read_csv(file_clientes, sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeDecodeError:
            file_clientes.seek(0)
            df_temp = pd.read_csv(file_clientes, sep=None, engine='python', encoding='iso-8859-1')
        
        # Limpa espaços invisíveis dos títulos (ex: "Sexo " vira "Sexo")
        df_temp.columns = df_temp.columns.astype(str).str.strip()
        
        # Força o filtro das colunas necessárias após a limpeza
        try:
            df_clientes = df_temp[colunas_clientes]
        except KeyError:
            # Se o separador automático falhar por completo, tenta forçar por ponto e vírgula (Padrão Excel BR)
            file_clientes.seek(0)
            try:
                df_temp = pd.read_csv(file_clientes, sep=';', encoding='utf-8-sig')
            except UnicodeDecodeError:
                file_clientes.seek(0)
                df_temp = pd.read_csv(file_clientes, sep=';', encoding='iso-8859-1')
            df_temp.columns = df_temp.columns.astype(str).str.strip()
            df_clientes = df_temp[colunas_clientes]
    
    # Limpeza estrita de nulos nas colunas fundamentais
    df_clientes = df_clientes.dropna(subset=['Nome', 'Data de nascimento', 'Sexo'])
    df_clientes['Nome'] = df_clientes['Nome'].astype(str).str.strip()
    
    # Conversão de data e cálculo de idade
    df_clientes['Data de nascimento'] = pd.to_datetime(df_clientes['Data de nascimento'], errors='coerce')
    df_clientes = df_clientes.dropna(subset=['Data de nascimento'])
    df_clientes['Idade'] = df_clientes['Data de nascimento'].apply(calcular_idade)
    
    # 2. Tratamento da Catraca (Mapeando 'Cliente' para 'Nome')
    colunas_catraca = ['Cliente', 'Contrato', 'Data']
    
    # Verifica a extensão para ler CSV ou Excel de forma resiliente
    if hasattr(file_catraca, 'filename') and file_catraca.filename.endswith('.csv'):
        try:
            df_catraca = pd.read_csv(file_catraca, sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeDecodeError:
            file_catraca.seek(0)
            df_catraca = pd.read_csv(file_catraca, sep=None, engine='python', encoding='iso-8859-1')
        df_catraca.columns = df_catraca.columns.astype(str).str.strip()
        df_catraca = df_catraca[colunas_catraca]
    else:
        df_temp_catraca = pd.read_excel(file_catraca)
        df_temp_catraca.columns = df_temp_catraca.columns.astype(str).str.strip()
        df_catraca = df_temp_catraca[colunas_catraca]
        
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
            {"nome": "Renan Santos (Simulado)", "probabilidade": 0.14},
            {"nome": "Claudio Ferreira (Simulado)", "probabilidade": 0.1523},
            {"nome": "Fernando Costa (Simulado)", "probabilidade": 0.9241}
        ]
    
    return executar_predicao_real(file_clientes, file_catraca)