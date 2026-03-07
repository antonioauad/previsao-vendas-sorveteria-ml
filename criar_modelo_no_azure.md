Passo a passo para criar o modelo no Azure

Passo 1: Criar o Workspace no Azure
1.	Acesse o Portal do Azure (você pode usar uma conta gratuita se ainda não tiver).
2.	Na barra de pesquisa, digite Azure Machine Learning e clique no serviço.
3.	Clique em Criar > Novo Workspace.
4.	Preencha os detalhes básicos (Nome do Workspace, Grupo de Recursos, Região) e clique em Revisar + Criar.
5.	Após a implantação, clique em Ir para o recurso e depois no botão Iniciar o Studio (Launch Studio).
Passo 2: Criar uma Instância de Computação (Sua máquina na nuvem)
Para rodar o código, precisamos de um servidor configurado.
1.	No menu lateral esquerdo do Azure ML Studio, vá em Compute (Computação).
2.	Na aba Compute instances, clique em New (Novo).
3.	Dê um nome à sua máquina, escolha o tipo de máquina virtual (uma CPU básica e barata, como a Standard_DS11_v2, é mais que suficiente para este projeto).
4.	Clique em Create. Em poucos minutos, ela estará com o status "Running" (Em execução).
Passo 3: Preparar os Arquivos no Studio
1.	No menu lateral esquerdo, vá em Notebooks.
2.	Você verá um explorador de arquivos. Crie uma pasta chamada projeto_sorveteria.
3.	Dentro dela, faça o upload do seu arquivo dataset_vendas.csv e crie um novo arquivo chamado modelo_vendas.ipynb (um Jupyter Notebook é perfeito para testar no Azure).

Passo 4: O Código adaptado para o Azure
Abra o seu modelo_vendas.ipynb no Azure ML Studio. Quando rodamos o código diretamente em uma Instância de Computação do Azure, ele já entende automaticamente que o MLflow deve registrar as métricas lá dentro.
Copie e cole este código em uma célula e execute:
Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# O Azure ML já configura o tracking URI do MLflow por baixo dos panos!
mlflow.set_experiment("Previsao_Vendas_Gelato_Magico")

with mlflow.start_run() as run:
    # 1. Carregar os dados (ajuste o caminho se necessário)
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

    # 5. Avaliar e registrar métricas
    mse = mean_squared_error(y_test, previsoes)
    r2 = r2_score(y_test, previsoes)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # Registrando no Azure ML via MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(modelo, "modelo_regressao_linear")
    
    print(f"Run ID: {run.info.run_id}")
    print("Modelo treinado e registrado com sucesso no Azure ML!")

Passo 5: Resultado 
Depois de rodar o código, vá até o menu lateral esquerdo e clique em Jobs (Trabalhos).
Você verá o seu experimento Previsao_Vendas_Gelato_Magico. 
Ao clicar nele, você terá um painel visual lindo mostrando:
•	Quem rodou o código.
•	As métricas (MSE e R2).
•	O modelo empacotado e pronto para ser consumido por uma aplicação de vendas ou um dashboard de TI.

