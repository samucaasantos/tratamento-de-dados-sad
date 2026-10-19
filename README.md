# FootDecision SAD — Pipeline de Engenharia de Dados e Machine Learning

> **Módulo de Ciência de Dados e Modelagem Analítica (Fase 1)** do sistema **FootDecision SAD (ScoutDecision Engine)** — um Sistema de Apoio à Decisão (SAD) voltado à otimização de orçamento de transferências e prospecção tática de atletas de futebol.

---

## 1. Visão Geral e Objetivo do Repositório

Este repositório concentra todo o pipeline de **Extração, Transformação, Harmonização (ETL) e Modelagem de Machine Learning** do projeto **FootDecision SAD**. O objetivo central é consolidar quatro fontes heterogêneas de dados esportivos, médicos e financeiros em uma base relacional íntegra, livre de valores nulos obrigatórios e enriquecida por inferências preditivas.

### Para que servem os dados tratados?

Os artefatos gerados na pasta [`result/`](./result) alimentam diretamente a arquitetura do SAD (Backend **Spring Boot 3** + Banco **PostgreSQL / H2** + Frontend **Next.js**), viabilizando quatro pilares de decisão:

1. **Simulação Dinâmica *What-If* e *Hard Filters*:**
   - Permite ao **Diretor Executivo (`ROLE_DIRECTOR`)** e ao **Analista de Mercado (`ROLE_ANALYST`)** filtrar atletas instantaneamente por posição (`GK`, `DF`, `MF`, `FW`), teto de transferência (`market_value`), teto salarial mensal (`salary`) e tolerância máxima a risco clínico (`injury_risk`).
2. **Motor de Similaridade Tática Contextual (Bifurcado):**
   - Utiliza os grupos (*clusters*) e centroides gerados pelos modelos **K-Means** (separados entre jogadores de linha e goleiros) para calcular a distância euclidiana inversa normalizada e encontrar substitutos com o perfil funcional exato exigido pelo elenco.
3. **Cálculo de Oportunidade Econômica (*Fair Market Value*):**
   - Compara o valor real de mercado com o **Valor Justo de Mercado** estimado pelo modelo de **Regressão Regularizada (Ridge)**, identificando atletas subvalorizados com alto retorno técnico por euro investido.
4. **Pontuação Multicritério (*Weighted Sum Model* - WSM) e Dossiê Visual:**
   - Alimenta a equação global de ranqueamento do SAD:
     $$\text{Score} = (w_1 \cdot \text{SimilaridadeTatica}) + (w_2 \cdot \text{OportunidadeEconomica}) + (w_3 \cdot (1 - \text{RiscoLesao}))$$
   - Fornece as métricas padronizadas por 90 minutos ($p90$) para renderização dos **Gráficos Radar** e emissão do parecer textual determinístico para o **Top 5** de recomendações.

---

## 2. Árvore de Arquivos do Projeto

```text
Tratamento-de-dados-sad/
├── .gitignore                              # Exclusão de bases brutas >100MB, caches e ambientes virtuais
├── README.md                               # Documentação técnica e guia de execução do pipeline
│
├── injuries/                               # [Entrada] Base histórica de lesões e afastamentos médicos
│   └── full_dataset_thesis - 1.csv         # Registros clínicos de dias parados e jogos perdidos
│
├── player-stats/                           # [Entrada] Estatísticas de desempenho em campo (FBref 26/27)
│   ├── players_data-2026_2027.csv          # Base completa das Big 5 ligas europeias
│   └── players_data_light-2026_2027.csv    # Base otimizada utilizada na ingestão do pipeline
│
├── transfermarket/                         # [Entrada - Ignorada no Git] Dumps brutos do Transfermarkt (~907 MB)
│   ├── players.csv                         # Dados cadastrais e valor de mercado (market_value_in_eur)
│   ├── player_valuations.csv               # Histórico de avaliações financeiras
│   ├── transfers.csv                       # Histórico de transferências
│   ├── appearances.csv                     # Participações em partidas (198 MB)
│   ├── game_events.csv                     # Eventos de jogo (179 MB)
│   ├── game_lineups.csv                    # Escalações (416 MB)
│   ├── games.csv                           # Registro de partidas
│   ├── club_games.csv                      # Estatísticas por clube
│   ├── clubs.csv                           # Cadastro de clubes
│   ├── competitions.csv                    # Cadastro de competições
│   ├── countries.csv                       # Cadastro de países
│   └── national_teams.csv                  # Seleções nacionais
│
├── wages/                                  # [Entrada] Folha salarial e dados contratuais (Capology)
│   └── dfAll.csv                           # Remunerações semanais/anuais em Euros e métricas auxiliares
│
├── scripts/                                # [Processamento] Scripts Python de Engenharia e Ciência de Dados
│   ├── process_data.py                     # Etapa 1: Limpeza, Fuzzy Matching, corte de minutagem e cálculo p90
│   ├── train_ml_models.py                  # Etapa 2: Treinamento de K-Means, Ridge Regression e Random Forest
│   └── export_sql.py                       # Etapa 3: Geração automatizada do script relacional data.sql
│
└── result/                                 # [Saída] Artefatos consolidados prontos para o Backend e Banco
    ├── players_cleaned.csv                 # Tabela 'players' higienizada (8 colunas do schema PostgreSQL)
    ├── player_metrics_cleaned.csv          # Tabela 'player_metrics' higienizada (14 colunas do schema PostgreSQL)
    ├── players_analyzed.csv                # Base analítica expandida (inclui Fair Market Value e lesões brutas)
    ├── models_metadata.json                # Centroides, escalonadores, pesos e métricas para o motor Java
    └── data.sql                            # Script DML (INSERTs) pronto para carga no PostgreSQL (Neon) ou H2
```

---

## 3. Pipeline de Tratamento e Engenharia de Atributos

O script [`scripts/process_data.py`](./scripts/process_data.py) executa as seguintes regras metodológicas estritas:

1. **Normalização e *Entity Resolution* (*Fuzzy Matching*):**
   - Padronização textual (remoção de diacríticos via `NFKD`, sufixos geracionais e pontuações).
   - Cruzamento exato e difuso (`RapidFuzz` com limiar rigoroso $\ge 92\%$) com desambiguação por clube e ano de nascimento.
2. **Regra Estrita de Amostragem e Completude:**
   - **Filtro de Minutagem:** Inclusão exclusiva de atletas com minutagem acumulada entre **450 e 900 minutos** (`450 <= Min <= 900`), estabilizando as taxas por 90 minutos ($p90$).
   - **Política de Tolerância Zero a Nulos (*Inner Join* Restritivo):** Descarte sumário de registros sem correspondência salarial (`salary`) ou valor de mercado (`market_value`).
   - **Funil de Dados:** `2.105` atletas (FBref bruto) $\rightarrow$ `251` elegíveis por minutagem $\rightarrow$ **`145` atletas validados e 100% íntegros** (110 de linha e 35 goleiros).
3. **Engenharia de Atributos Bifurcada ($p90$):**
   - **Atletas de Linha (`Pos != 'GK'`):** `progressive_passes_p90`, `tackles_p90`, `sca_p90`, `progressive_carries_p90`, `expected_goals_p90`, `expected_assists_p90` (colunas de goleiro permanecem `NULL`).
   - **Goleiros (`Pos == 'GK'`):** `goals_conceded_p90`, `save_percentage`, `clean_sheet_percentage`, `shots_on_target_against_p90`, `penalties_saved_p90` (colunas de linha permanecem `NULL`).

---

## 4. Modelos de Machine Learning Implementados

O script [`scripts/train_ml_models.py`](./scripts/train_ml_models.py) treina quatro modelos especializados e persiste seus hiperparâmetros e métricas em [`result/models_metadata.json`](./result/models_metadata.json):

| Modelo | Algoritmo | Variáveis de Entrada | Resultado / Métricas de Validação |
| :--- | :--- | :--- | :--- |
| **Agrupamento Tático 1 (Linha)** | `K-Means` ($K=3$) + `StandardScaler` | `progressive_passes_p90`, `tackles_p90`, `sca_p90`, `progressive_carries_p90` | **Silhouette Score: `0.2736`**<br>• Cluster 1: Marcador / Destruidor<br>• Cluster 2: Construtor / Passador Base<br>• Cluster 3: Criador / Transição Ofensiva |
| **Agrupamento Tático 2 (Goleiros)** | `K-Means` ($K=3$) + `StandardScaler` | `goals_conceded_p90`, `save_percentage`, `clean_sheet_percentage`, `shots_on_target_against_p90`, `penalties_saved_p90` | **Silhouette Score: `0.3581`**<br>• Cluster 4: Exposto / Sob Pressão<br>• Cluster 5: Muro / Titular Seguro<br>• Cluster 6: Especialista em Pênaltis |
| **Precificação (*Fair Market Value*)** | `RidgeCV` (L2 Regularizada com alvo logarítmico) | `age`, `minutes_played`, *dummies* de posição e métricas funcionais $p90$ | **$\alpha = 9.1030$**<br>• **$R^2$ (Espaço Log): `0.3389`**<br>• **$R^2$ (Euros): `0.1621`** \| **RMSE: `€ 31,5M`** |
| **Classificador de Risco Médico** | `RandomForestClassifier` & `DecisionTreeClassifier` | `total_days_injured`, `total_games_missed`, `injury_count`, `age`, `is_goalkeeper` | **F1-Score (Macro & Weighted): `1.0000`**<br>Saída discreta: `Baixo` (73), `Moderado` (33), `Alto` (39) |

---

## 5. Schema Relacional Gerado (`result/data.sql`)

O script [`scripts/export_sql.py`](./scripts/export_sql.py) converte os CSVs finais em instruções SQL compatíveis com o modelo relacional 1:1 do PostgreSQL:

- **Tabela `players` ([`players_cleaned.csv`](./result/players_cleaned.csv)):**
  - `id`, `name`, `position`, `age`, `current_club`, `market_value`, `salary`, `injury_risk`
- **Tabela `player_metrics` ([`player_metrics_cleaned.csv`](./result/player_metrics_cleaned.csv)):**
  - `player_id`, `minutes_played`, `cluster_id`, `progressive_passes_p90`, `tackles_p90`, `sca_p90`, `progressive_carries_p90`, `expected_goals_p90`, `expected_assists_p90`, `goals_conceded_p90`, `save_percentage`, `clean_sheet_percentage`, `shots_on_target_against_p90`, `penalties_saved_p90`

---

## 6. Como Executar o Pipeline Localmente

### Pré-requisitos
- **Python 3.10+**
- Bibliotecas listadas abaixo:
  ```bash
  pip install pandas numpy scikit-learn rapidfuzz
  ```

### Passo a Passo de Execução

Na raiz do repositório, execute os três scripts na ordem do pipeline:

```bash
# 1. Limpeza, cruzamento de entidades e cálculo de métricas p90
python scripts/process_data.py

# 2. Treinamento dos modelos de ML (K-Means, Ridge e Random Forest) e geração de metadados
python scripts/train_ml_models.py

# 3. Exportação do arquivo relacional result/data.sql para o banco de dados
python scripts/export_sql.py
```
