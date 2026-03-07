# previsao-vendas-sorveteria-ml
Modelo de previsão de vendas de sorvetes conforme a temperatura para evitar perdas
# 🍦 Previsão de Vendas de Sorvete com Machine Learning

## 📖 Entendendo o Desafio
Este projeto foi desenvolvido como parte de um desafio prático da plataforma DIO. O objetivo é ajudar a sorveteria fictícia **Gelato Mágico** a otimizar sua produção diária. Como as vendas de sorvete em uma cidade litorânea possuem forte correlação com a temperatura, o uso de Machine Learning permite antecipar a demanda, evitando prejuízos com desperdícios (produção excessiva) ou perda de vendas (falta de produto).

## 🚀 Tecnologias Utilizadas
* **Python:** Linguagem principal para análise e modelagem de dados.
* **Scikit-Learn:** Biblioteca para criação do modelo de Regressão Linear.
* **MLflow:** Ferramenta para gerenciar o ciclo de vida do modelo, registrando métricas de treino e versionando o algoritmo.
* **Pandas:** Para manipulação e leitura do dataset (`inputs/dataset_vendas.csv`).

## 🧠 O Processo e Insights
1. **Coleta de Dados:** Simulamos um dataset contendo registros históricos de temperatura (em Celsius) e a quantidade correspondente de sorvetes vendidos.
2. **Treinamento do Modelo:** Utilizamos um modelo de Regressão Linear, ideal para entender a relação contínua entre temperatura e vendas.
3. **Gerenciamento com MLflow:** O MLflow foi integrado para registrar o erro quadrático médio (MSE) e o coeficiente de determinação (R²), garantindo que futuras melhorias no modelo possam ser comparadas com esta versão inicial (Reprodutibilidade).
4. **Insights de Negócios:** * A correlação positiva forte indica que os esforços de marketing e equipe de vendas devem ser escalados nos dias em que a previsão do tempo aponta temperaturas acima de 30°C.
   * Nos dias mais frios, a operação pode atuar com equipe reduzida, otimizando custos operacionais (TI e Vendas alinhados com a eficiência financeira).

## ☁️ Possibilidades Futuras (Deploy em Cloud)
O próximo passo lógico para este projeto é a sua produtização. A implementação pode ser feita utilizando a infraestrutura em nuvem, como o **Microsoft Azure Machine Learning**, permitindo:
* Que o sistema do caixa da sorveteria consuma a previsão de vendas via API REST.
* O acionamento de alertas automáticos para o estoque e produção com base na previsão meteorológica da semana.
* Um dashboard interativo (Power BI) para acompanhamento gerencial.

## 📁 Estrutura do Repositório
* `/inputs`: Contém o dataset utilizado no projeto (`dataset_vendas.csv`).
* `/src`: Scripts de treinamento e gerenciamento do MLflow.
* `README.md`: Documentação e racional do projeto.
