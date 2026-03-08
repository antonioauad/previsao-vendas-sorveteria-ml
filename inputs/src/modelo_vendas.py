import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import joblib # Importante: Biblioteca nativa para salvar modelos

# O Azure ML já configura o tracking URI do MLflow por baixo dos panos!
mlflow.set_experiment("Previsao_Vendas_Gelato_Magico")

with mlflow.start_run() as run:
    # 1. Carregar os dados
    df = pd.read_csv('dataset_vendas.csv')
    
    X = df[['temperatura_celsius']]
    y = df['sorvetes_vendidos']

    # 2. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 4. Fazer previsões
    previsoes = modelo.predict(X_test)

    # 5. Avaliar e registrar métricas no MLflow
    mse = mean_squared_error(y_test, previsoes)
    r2 = r2_score(y_test, previsoes)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    
    # 6. A SOLUÇÃO: Salvar o modelo localmente e enviar como artefato
    nome_arquivo = "modelo_regressao_linear.pkl"
    joblib.dump(modelo, nome_arquivo)
    mlflow.log_artifact(nome_arquivo)
    
    print(f"Run ID: {run.info.run_id}")
    print("Modelo treinado e salvo como artefato com sucesso no Azure ML! 🚀")
