import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Iniciar o rastreamento do MLflow
mlflow.set_experiment("Previsao_Vendas_Gelato_Magico")

with mlflow.start_run():
    # 1. Carregar os dados
    df = pd.read_csv('../inputs/dataset_vendas.csv')
    
    X = df[['temperatura_celsius']]
    y = df['sorvetes_vendidos']

    # 2. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 4. Fazer previsões
    previsoes = modelo.predict(X_test)

    # 5. Avaliar e registrar métricas
    mse = mean_squared_error(y_test, previsoes)
    r2 = r2_score(y_test, previsoes)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(modelo, "modelo_regressao_linear")
    
    print("Modelo treinado e registrado com sucesso no MLflow!")
