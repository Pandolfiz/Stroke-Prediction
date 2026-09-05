# Predição de AVC (Stroke) — Análise Exploratória e Modelagem com Machine Learning

Notebook de análise e predição de ocorrência de AVC (Acidente Vascular Cerebral) a partir de dados clínicos e demográficos, com EDA, três modelos de classificação, comparação de estratégias de balanceamento, otimização de hiperparâmetros e explicabilidade via SHAP.

## Dataset

- Fonte: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, `fedesoriano/stroke-prediction-dataset`)
- Baixado automaticamente pelo notebook via `kagglehub`
- ~5.110 registros, variável-alvo `stroke` (0/1), fortemente desbalanceada (~5% de casos positivos)
- Variáveis: gênero, idade, hipertensão, doença cardíaca, estado civil, tipo de trabalho, tipo de residência, nível médio de glicose, IMC (BMI) e status de fumante

## Estrutura do notebook

**1. Análise Exploratória (EDA)**
- Distribuição da variável-alvo e das variáveis explicativas
- Testes estatísticos (Mann-Whitney U, qui-quadrado) relacionando idade, glicose e BMI à ocorrência de AVC
- Tratamento de valores ausentes em `bmi` (imputação pela mediana + flag de ausência)
- Visualizações: idade × glicose × BMI, proporção acumulada de AVC por idade, entre outras

**2. Preparação para Machine Learning**
- Codificação de variáveis categóricas (`gender`, `Residence_type`, `work_type`, `ever_married`, `smoking_status`)
- Split treino/teste estratificado (80/20)

**3. Modelagem — 3 algoritmos**
- Logistic Regression, Random Forest e XGBoost
- Duas estratégias de balanceamento comparadas lado a lado:
  - **SMOTE** (reamostragem sintética da classe minoritária no treino)
  - **class_weight / scale_pos_weight** (penalização da classe minoritária, sem reamostragem)

**4. Otimização de hiperparâmetros**
- `GridSearchCV` com validação cruzada estratificada (5 folds) para os 3 algoritmos
- SMOTE aplicado *dentro* de cada fold via `imblearn.pipeline.Pipeline`, evitando vazamento de dados (Dataleake)
- Métrica de otimização: `average_precision` (PR-AUC), mais adequada que ROC-AUC para dados desbalanceados

**5. Avaliação**
- Métricas: Accuracy, Precision, Recall, F1 e ROC-AUC para as 9 combinações modelo × estratégia
- Curvas Precision-Recall e Average Precision por modelo
- Threshold ótimo (que maximiza F1) comparado ao threshold padrão de 0.5, com matrizes de confusão lado a lado

**6. Explicabilidade com SHAP**
- `TreeExplainer` aplicado ao Random Forest e ao XGBoost (melhor estratégia de cada, segundo ROC-AUC)
- Summary plots de importância das variáveis
- Comparação da importância média (|SHAP value|) entre os dois modelos
- Explicação individual (waterfall plot) do caso com maior probabilidade prevista de AVC

## Requisitos

```
pandas
numpy
matplotlib
seaborn
plotly
kagglehub
imbalanced-learn
scikit-learn
xgboost
shap
scipy
```

Instalação:

```bash
pip install pandas numpy matplotlib seaborn plotly kagglehub imbalanced-learn scikit-learn xgboost shap scipy
```

## Como rodar

1. Configure suas credenciais do Kaggle (necessárias para o `kagglehub` baixar o dataset — veja a [documentação do kagglehub](https://github.com/Kaggle/kagglehub) para gerar seu `kaggle.json`)
2. Instale as dependências acima
3. Execute `Analise.ipynb` célula a célula, do início ao fim

## Observações

- O dataset é fortemente desbalanceado (~5% de casos de AVC), por isso Accuracy sozinha não é um bom critério de avaliação — o notebook prioriza Recall, F1, ROC-AUC e PR-AUC
- Os resultados exatos de cada execução (métricas, melhores hiperparâmetros, threshold ótimo) podem variar de acordo com a versão dos dados baixados e a aleatoriedade dos algoritmos, mesmo com `random_state` fixado em alguns pontos
