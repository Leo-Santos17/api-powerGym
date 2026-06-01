import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Importa o runtime do TFLite de forma flexível (usa o leve ou o cheio, o que estiver instalado)
import ai_edge_litert.interpreter as tflite

# -------------------------------------------------------------------------
# INICIALIZAÇÃO: Carrega os modelos globais leves em memória
# -------------------------------------------------------------------------
MODEL_TFLITE_PATH = 'modelo_churn.tflite'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

interpreter = None
input_details = None
output_details = None
scaler = None
label_encoder = None

if os.path.exists(MODEL_TFLITE_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    # Inicializa o Interpretador TF Lite (Consumo mínimo de RAM)
    interpreter = tflite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # MAPEIA AS ENTRADAS E SAÍDAS DO MODELO
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("--- MODELOS LEVES TF-LITE CARREGADOS COM SUCESSO ---")
else:
    print("--- AVISO: Arquivos .tflite ou .pkl não encontrados. Rodando modo simulação. ---")


def calcular_idade(nascimento):
    if pd.isnull(nascimento):
        return None
    hoje = datetime.now()
    return hoje.year - nascimento.year - ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))


def executar_predicao_real(file_clientes, file_catraca):
    """
    Executa o pipeline completo de tratamento estrito e predição usando TF Lite.
    """
    # 1. Tratamento da Base de Clientes (Híbrido: Excel ou CSV)
    colunas_clientes = ['Nome', 'Situação do contrato', 'Situação do cliente', 'Data de nascimento', 'Sexo']

    if hasattr(file_clientes, 'filename') and (file_clientes.filename.endswith('.xlsx') or file_clientes.filename.endswith('.xls')):
        df_temp = pd.read_excel(file_clientes)
        df_temp.columns = df_temp.columns.astype(str).str.strip()
        df_clientes = df_temp[colunas_clientes]
    else:
        try:
            df_temp = pd.read_csv(file_clientes, sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeDecodeError:
            file_clientes.seek(0)
            df_temp = pd.read_csv(file_clientes, sep=None, engine='python', encoding='iso-8859-1')
        
        df_temp.columns = df_temp.columns.astype(str).str.strip()
        
        try:
            df_clientes = df_temp[colunas_clientes]
        except KeyError:
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
    
    df_catraca = df_catraca.dropna(subset=['Nome', 'Data'])
    df_catraca['Nome'] = df_catraca['Nome'].astype(str).str.strip()
    
    termos_invalidos = ['CANCELADO', 'diaria 7', 'nan', '']
    df_catraca = df_catraca[~df_catraca['Nome'].isin(termos_invalidos)]
    
    df_catraca['Data'] = pd.to_datetime(df_catraca['Data'], errors='coerce')
    df_catraca = df_catraca.dropna(subset=['Data'])
    
    # 3. Merge e Agrupamento
    base_mista = df_catraca.merge(df_clientes[['Nome', 'Sexo', 'Idade']], on='Nome', how='inner')
    
    if base_mista.empty:
        return {"error": "Nenhum dado compatível encontrado após a limpeza das planilhas."}
        
    df_features = base_mista.groupby('Nome').agg(
        Total_Visitas=('Data', 'count'),
        Idade=('Idade', 'first'),
        Sexo=('Sexo', 'first')
    ).reset_index()
    
    df_features = df_features[df_features['Sexo'].isin(label_encoder.classes_)]
    df_features['Sexo'] = label_encoder.transform(df_features['Sexo'].astype(str))
    
    # 4. Preparação da Matriz e Escalonamento
    X_raw = df_features[['Total_Visitas', 'Idade', 'Sexo']].values
    X_scaled = scaler.transform(X_raw).astype(np.float32) # TF Lite exige float32 explícito
    
    lista_nomes = df_features['Nome'].tolist()
    resultado = []
    
    # 5. Inferência com o TF Lite (Processamento por Aluno / Batch)
    for i, linha_input in enumerate(X_scaled):
        # Adequa o shape da linha para bater com o esperado [1, num_features]
        dados_aluno = np.expand_dims(linha_input, axis=0)
        
        # Injeta os dados no tensor de input do interpretador
        interpreter.set_tensor(input_details[0]['index'], dados_aluno)
        
        # Executa a inferência
        interpreter.invoke()
        
        # Coleta a probabilidade calculada no tensor de saída
        probabilidade = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        resultado.append({
            "nome": lista_nomes[i],
            "probabilidade": float(probabilidade)
        })
        
    return resultado


def predict_churn(file_clientes=None, file_catraca=None):
    if file_clientes is None or file_catraca is None or interpreter is None:
        return [
            {"nome": "Ana Silva (Simulado)", "probabilidade": 0.8146},
            {"nome": "Claudio Ferreira (Simulado)", "probabilidade": 0.1523},
            {"nome": "Fernando Costa (Simulado)", "probabilidade": 0.9241}
        ]
    
    return executar_predicao_real(file_clientes, file_catraca)