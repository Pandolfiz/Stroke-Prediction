# Predição de AVC (Stroke) — Análise Exploratória e Modelagem com Machine Learning

Notebook de análise e predição de AVC (Acidente Vascular Cerebral) a partir de dados clínicos e demográficos. Inclui EDA com testes estatísticos, três modelos de classificação com duas estratégias de balanceamento, otimização de hiperparâmetros, interpretabilidade (coeficientes e SHAP), calibração das probabilidades e intervalos de confiança para as métricas.

Todas as escolhas (pré-processamento, hiperparâmetros, modelo e threshold) usam **somente os dados de treino**. O conjunto de teste só é usado para a avaliação final.

## Dataset

- Fonte: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, `fedesoriano/stroke-prediction-dataset`)
- O notebook baixa o dataset automaticamente com o `kagglehub`
- 5.110 registros. A variável-alvo `stroke` (0/1) é fortemente desbalanceada: ~4,9% de casos positivos
- Variáveis: gênero, idade, hipertensão, doença cardíaca, estado civil, tipo de trabalho, tipo de residência, glicose média, IMC (BMI) e status de fumante
- O `bmi` tem 201 valores ausentes

## Estrutura do notebook

**1. Carregamento**
- Download e leitura dos dados e checagem de tipos e valores ausentes

**2. Análise Exploratória (EDA)**
- Distribuição de idade, glicose e BMI por ocorrência de AVC, com o teste de Mann-Whitney U
- Taxa de AVC por faixa etária, proporção acumulada de AVC e painel-resumo das variáveis
- Qui-quadrado para as variáveis categóricas
- Criação da flag `bmi_missing`. A ausência do BMI está fortemente associada ao AVC
- Cada resultado vem acompanhado de um markdown com a interpretação

**3. Machine Learning**
- **Codificação:** `ever_married` vira 0/1, e `work_type` e `smoking_status` viram one-hot (`dtype=int`)
- **Features:** `gender` e `Residence_type` ficam de fora, porque o qui-quadrado não mostrou associação com o AVC
- **Divisão:** treino/teste estratificado **70/30**
- **Pipeline (`imblearn.pipeline.Pipeline`):**
  - `SimpleImputer` (mediana do BMI)
  - `StandardScaler` (só na Logistic Regression)
  - SMOTE (opcional)
  - classificador

  Todo o pré-processamento é ajustado dentro de cada fold, o que evita vazamento de dados (*data leakage*).
- **Modelos:** Logistic Regression, Random Forest e XGBoost
- **Estratégias de balanceamento:**
  - **SMOTE**, com reamostragem sintética da classe minoritária
  - **`class_weight` / `scale_pos_weight`**, que penalizam o erro na classe minoritária sem reamostrar
- **GridSearchCV:** validação cruzada estratificada de 5 folds e `scoring="average_precision"` (PR-AUC), mais adequada que a ROC-AUC para dados desbalanceados
- **Avaliação:**
  - Accuracy, Precision, Recall, F1, ROC-AUC e curvas Precision-Recall para as 9 combinações de modelo e estratégia
- **Threshold:**
  - escolhido pelas previsões *out-of-fold* (`cross_val_predict`) do treino;
  - comparado ao threshold padrão de 0,5 por meio de matrizes de confusão

**4. Interpretabilidade**
- Coeficientes padronizados da Logistic Regression
- SHAP para o Random Forest (`TreeExplainer`) e para o XGBoost (TreeSHAP nativo, `pred_contribs=True`)
- Importância das variáveis em % do total, para comparar modelos que usam escalas diferentes
- Waterfall plot do paciente com maior probabilidade prevista de AVC

**5. Refinamento e validação**
- **5.1 Calibração:** `CalibratedClassifierCV` (sigmoid), avaliada por curva de calibração e Brier score
- **5.2 Threshold por F-beta:** F2 (β = 2), que dá mais peso ao recall, porque deixar de detectar um AVC custa mais que um alarme falso
- **5.3 Intervalos de confiança:**
  - validação cruzada repetida (5 folds × 10 repetições);
  - bootstrap pareado no teste (2.000 reamostragens), que gera ICs das métricas e das diferenças de ROC-AUC entre os modelos
- **5.4 Ablação de `bmi_missing`:** desempenho dos modelos com e sem a variável

**6. Conclusões**
- Fatores de risco, modelo final, comparação entre modelos, metodologia (por que só o treino foi usado nas escolhas), pontos de atenção e próximos passos

## Principais resultados

**Modelo final:** Logistic Regression (GridSearchCV + SMOTE, `C = 0,01`), calibrada e com threshold **0,075**, escolhido pelo F2.

| Métrica (teste) | Valor | IC 95% |
|---|---|---|
| ROC-AUC | 0,84 | 0,79 – 0,88 |
| Recall | 0,73 | 0,63 – 0,83 |
| Precisão | 0,17 | 0,13 – 0,21 |
| F2 | 0,44 | 0,36 – 0,51 |

- **Detecção:** o modelo detecta 55 dos 75 AVCs do teste e emite alertas para ~21% dos pacientes.
- **Calibração:** o risco médio previsto (4,9%) é igual à taxa real, e o Brier score cai de 0,166 para 0,041.
- **Fatores de risco:** os mais fortes são **idade**, **ausência de BMI**, **hipertensão** e **glicose**.
- **Comparação entre modelos:** a Logistic Regression foi igual ou melhor que Random Forest e XGBoost em todas as avaliações. Essas diferenças estão dentro da incerteza, porque os ICs das diferenças contêm zero. Ela foi escolhida pela simplicidade e pela interpretabilidade.
- **Atenção à variável `bmi_missing`:** sem ela, a average precision cai de 12% (Logistic Regression) a 27% (XGBoost). A variável reflete o processo de coleta, e não a biologia. Para uso preventivo, a estimativa realista é a do modelo sem ela: AP ~0,19 e ROC-AUC ~0,84.

## Requisitos

Dependências em [`requirements.txt`](requirements.txt):

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
xgboost
shap
kagglehub
```

Versões testadas:
- Python 3.10
- pandas 2.3
- numpy 2.2
- scikit-learn 1.7
- imbalanced-learn 0.14
- xgboost 3.2
- shap 0.49
- scipy 1.15
- kagglehub 1.0

Instalação:

```bash
pip install -r requirements.txt
```

## Como rodar

1. Instale as dependências acima, de preferência em um ambiente virtual.
2. Execute `Analise.ipynb` do início ao fim. O dataset é baixado na primeira célula de carregamento. Se o `kagglehub` pedir autenticação, configure suas credenciais do Kaggle conforme a [documentação do kagglehub](https://github.com/Kaggle/kagglehub).
3. A execução completa leva alguns minutos, principalmente por causa do GridSearchCV, da validação cruzada repetida e do bootstrap.

## Observações

- **Accuracy:** o dataset é muito desbalanceado, então a accuracy sozinha não serve como critério. O notebook prioriza average precision, recall, F1/F2 e ROC-AUC.
- **Reprodutibilidade:** todas as fontes de aleatoriedade usam `random_state=42`, e reexecuções produzem os mesmos resultados. Eles só devem mudar se o dataset baixado mudar ou se forem usadas versões muito diferentes das bibliotecas.
- **Compatibilidade do SHAP:** o `shap.TreeExplainer` não lê o `base_score` salvo pelo XGBoost 3.x. Por isso o notebook usa o TreeSHAP nativo do XGBoost, que calcula os mesmos valores.
- **Uso:** o projeto é educacional. O modelo não foi validado com dados externos e não deve ser usado para decisões clínicas.
