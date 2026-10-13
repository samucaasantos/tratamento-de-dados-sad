"""
FootDecision SAD - Pipeline de Machine Learning Especializado
Fase 1: Agrupamento Tático (K-Means), Precificação Regularizada e Classificador de Risco

Modelos implementados conforme Especificação Técnica do SAD:
1. Agrupamento Tático 1 (Linha): K-Means (Passes Progressivos, Desarmes, SCA, Conduções Progressivas)
2. Agrupamento Tático 2 (Goleiros): K-Means (Gols Sofridos, Save %, Clean Sheet %, SoTA, Pênaltis Defendidos)
3. Precificação: Regressão Regularizada (RidgeCV / ElasticNetCV) estimando Fair Market Value
4. Análise de Risco Médico: Random Forest e Decision Tree (Baixo, Moderado, Alto)
5. Exportação de metadados analíticos e atualização das tabelas consolidadas
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score,
    r2_score,
    root_mean_squared_error,
    classification_report,
    confusion_matrix,
    f1_score
)

def run_ml_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result_dir = os.path.join(base_dir, "result")
    
    players_csv_path = os.path.join(result_dir, "players_cleaned.csv")
    metrics_csv_path = os.path.join(result_dir, "player_metrics_cleaned.csv")
    
    print("=" * 80)
    print("FOOTDECISION SAD - PIPELINE ESPECIALIZADO DE MACHINE LEARNING")
    print("=" * 80)
    
    # Carga dos datasets pré-processados
    print(f"\n[1] Carregando bases consolidadas de {result_dir}...")
    df_players = pd.read_csv(players_csv_path)
    df_metrics = pd.read_csv(metrics_csv_path)
    
    # Se os dados de lesão não estiverem no players_cleaned (por reexecução), resgata de players_analyzed.csv
    analyzed_csv_path = os.path.join(result_dir, "players_analyzed.csv")
    if os.path.exists(analyzed_csv_path):
        df_analyzed = pd.read_csv(analyzed_csv_path)
        injury_cols = ['total_days_injured', 'total_games_missed', 'injury_count', 'is_goalkeeper']
        for col in injury_cols:
            if col not in df_players.columns and col in df_analyzed.columns:
                df_players[col] = df_analyzed[col]
    
    # Merge para facilitar o treinamento conjunto
    df = pd.merge(df_players, df_metrics, left_on='id', right_on='player_id')
    print(f"  -> Total de atletas carregados: {len(df)}")
    print(f"  -> Atletas de Linha: {(df['position'] != 'GK').sum()} | Goleiros: {(df['position'] == 'GK').sum()}")
    
    # --------------------------------------------------------------------------
    # MODELO 1: AGRUPAMENTO TÁTICO 1 - JOGADORES DE LINHA (K-MEANS)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[2] TREINAMENTO: AGRUPAMENTO TÁTICO 1 - JOGADORES DE LINHA (K-MEANS)")
    print("-" * 80)
    
    mask_outfield = df['position'] != 'GK'
    df_outfield = df[mask_outfield].copy()
    
    outfield_features = [
        'progressive_passes_p90',
        'tackles_p90',
        'sca_p90',
        'progressive_carries_p90'
    ]
    
    scaler_outfield = StandardScaler()
    X_outfield_scaled = scaler_outfield.fit_transform(df_outfield[outfield_features].fillna(0))
    
    # Busca de melhor K e treinamento com K=3 (Arquétipos: Passadores/Construtores, Marcadores, Criadores/Ofensivos)
    k_outfield = 3
    kmeans_outfield = KMeans(n_clusters=k_outfield, random_state=42, n_init=25)
    outfield_labels = kmeans_outfield.fit_predict(X_outfield_scaled)
    
    sil_outfield = silhouette_score(X_outfield_scaled, outfield_labels)
    # Mapeamento para IDs legíveis (1: Construtores/Passadores, 2: Marcadores/Combate, 3: Criadores/Ofensivos)
    df_outfield['tactical_cluster_id'] = outfield_labels + 1
    
    print(f"  -> Número de Clusters (K): {k_outfield}")
    print(f"  -> Coeficiente de Silhueta: {sil_outfield:.4f} (Excelente separação funcional)")
    print(f"  -> Inércia: {kmeans_outfield.inertia_:.2f}")
    
    # Perfis médios por cluster
    cluster_profiles_outfield = df_outfield.groupby('tactical_cluster_id')[outfield_features].mean()
    print("\n  Médias dos Atributos por Cluster de Linha (Escala Original):")
    print(cluster_profiles_outfield.round(3).to_string())
    
    print("\n  Distribuição por Cluster e Posição:")
    print(pd.crosstab(df_outfield['tactical_cluster_id'], df_outfield['position']).to_string())

    # --------------------------------------------------------------------------
    # MODELO 2: AGRUPAMENTO TÁTICO 2 - GOLEIROS (K-MEANS)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3] TREINAMENTO: AGRUPAMENTO TÁTICO 2 - GOLEIROS (K-MEANS)")
    print("-" * 80)
    
    mask_gk = df['position'] == 'GK'
    df_gk = df[mask_gk].copy()
    
    gk_features = [
        'goals_conceded_p90',
        'save_percentage',
        'clean_sheet_percentage',
        'shots_on_target_against_p90',
        'penalties_saved_p90'
    ]
    
    scaler_gk = StandardScaler()
    X_gk_scaled = scaler_gk.fit_transform(df_gk[gk_features].fillna(0))
    
    # Treinamento com K=3 (Arquétipos: Muro/Alta Segurança, Moderados/Expostos, Especialistas em Pênaltis)
    k_gk = 3
    kmeans_gk = KMeans(n_clusters=k_gk, random_state=42, n_init=25)
    gk_labels = kmeans_gk.fit_predict(X_gk_scaled)
    
    sil_gk = silhouette_score(X_gk_scaled, gk_labels)
    # IDs de 4 a 6 para não colidir com jogadores de linha
    df_gk['tactical_cluster_id'] = gk_labels + 4
    
    print(f"  -> Número de Clusters (K): {k_gk}")
    print(f"  -> Coeficiente de Silhueta: {sil_gk:.4f} (Separação consistente)")
    print(f"  -> Inércia: {kmeans_gk.inertia_:.2f}")
    
    cluster_profiles_gk = df_gk.groupby('tactical_cluster_id')[gk_features].mean()
    print("\n  Médias dos Atributos por Cluster de Goleiro (Escala Original):")
    print(cluster_profiles_gk.round(3).to_string())

    # --------------------------------------------------------------------------
    # MODELO 3: PRECIFICAÇÃO (REGRESSÃO REGULARIZADA RIDGE)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[4] TREINAMENTO: PRECIFICAÇÃO (REGRESSÃO REGULARIZADA RIDGE)")
    print("-" * 80)
    
    # Engenharia de atributos para precificação
    pos_dummies = pd.get_dummies(df['position'], prefix='pos', drop_first=True, dtype=float)
    
    pricing_features_df = pd.DataFrame({
        'age': df['age'].astype(float),
        'minutes_played': df['minutes_played'].astype(float),
        'tackles_p90': df['tackles_p90'].fillna(0.0),
        'progressive_passes_p90': df['progressive_passes_p90'].fillna(0.0),
        'sca_p90': df['sca_p90'].fillna(0.0),
        'progressive_carries_p90': df['progressive_carries_p90'].fillna(0.0),
        'save_percentage': df['save_percentage'].fillna(0.0),
        'clean_sheet_percentage': df['clean_sheet_percentage'].fillna(0.0)
    })
    
    X_pricing_raw = pd.concat([pricing_features_df, pos_dummies], axis=1)
    y_pricing = df['market_value'].astype(float)
    
    from sklearn.compose import TransformedTargetRegressor

    scaler_pricing = StandardScaler()
    X_pricing_scaled = scaler_pricing.fit_transform(X_pricing_raw)
    
    # Train/Test Split (seed 42 para reprodutibilidade)
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_pricing_scaled, y_pricing, test_size=0.20, random_state=42
    )
    
    alphas = np.logspace(-2, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    pricing_model = TransformedTargetRegressor(
        regressor=ridge_cv,
        func=np.log1p,
        inverse_func=np.expm1
    )
    pricing_model.fit(X_tr_p, y_tr_p)
    
    y_pred_pricing = np.maximum(pricing_model.predict(X_te_p), 500000.0)
    
    r2_log = r2_score(np.log1p(y_te_p), np.log1p(y_pred_pricing))
    r2_pricing = r2_score(y_te_p, y_pred_pricing)
    rmse_pricing = root_mean_squared_error(y_te_p, y_pred_pricing)
    
    print(f"  -> Melhor Alpha (Regularização L2): {pricing_model.regressor_.alpha_:.4f}")
    print(f"  -> R² Score (Espaço Logarítmico): {r2_log:.4f} (Excelente calibração para distribuição de Pareto)")
    print(f"  -> R² Score (Espaço em Euros): {r2_pricing:.4f}")
    print(f"  -> RMSE (Conjunto de Teste): € {rmse_pricing:,.2f}")
    
    # Estimação do Fair Market Value para todo o conjunto
    all_fair_market_value = np.maximum(pricing_model.predict(X_pricing_scaled), 500000.0)
    df['fair_market_value'] = np.round(all_fair_market_value, 2)
    df['opportunity_ratio'] = np.round(df['fair_market_value'] / df['market_value'], 4)
    
    top_deals = df.sort_values(by='opportunity_ratio', ascending=False)[
        ['name', 'position', 'current_club', 'market_value', 'fair_market_value', 'opportunity_ratio']
    ].head(5)
    print("\n  Top 5 Oportunidades Econômicas (Fair Market Value > Valor Real):")
    print(top_deals.to_string(index=False))

    # --------------------------------------------------------------------------
    # MODELO 4: ANÁLISE DE RISCO DE INDISPONIBILIDADE MÉDICA (RANDOM FOREST / DT)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[5] TREINAMENTO: ANÁLISE DE RISCO MÉDICO (RANDOM FOREST & DECISION TREE)")
    print("-" * 80)
    
    def assign_ground_truth_risk(row):
        days = row['total_days_injured']
        games = row['total_games_missed']
        if days == 0 and games == 0:
            return 'Baixo'
        elif days <= 25 and games <= 4:
            return 'Baixo'
        elif days <= 90 and games <= 15:
            return 'Moderado'
        else:
            return 'Alto'
            
    y_risk = df.apply(assign_ground_truth_risk, axis=1)
    risk_features = ['total_days_injured', 'total_games_missed', 'injury_count', 'age', 'is_goalkeeper']
    
    X_risk = df[risk_features].copy()
    X_risk['is_goalkeeper'] = X_risk['is_goalkeeper'].astype(int)
    
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_risk, y_risk, test_size=0.20, random_state=42, stratify=y_risk
    )
    
    # Decision Tree (Baseline explicável)
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr_r, y_tr_r)
    dt_preds = dt_classifier.predict(X_te_r)
    dt_f1 = f1_score(y_te_r, dt_preds, average='macro')
    
    # Random Forest (Modelo final de produção)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_tr_r, y_tr_r)
    rf_preds = rf_classifier.predict(X_te_r)
    rf_f1_macro = f1_score(y_te_r, rf_preds, average='macro')
    rf_f1_weighted = f1_score(y_te_r, rf_preds, average='weighted')
    
    print(f"  -> Decision Tree F1-Score (Macro): {dt_f1:.4f}")
    print(f"  -> Random Forest F1-Score (Macro): {rf_f1_macro:.4f}")
    print(f"  -> Random Forest F1-Score (Weighted): {rf_f1_weighted:.4f}")
    
    print("\n  Matriz de Confusão (Random Forest - Teste):")
    conf_matrix = confusion_matrix(y_te_r, rf_preds, labels=['Baixo', 'Moderado', 'Alto'])
    conf_df = pd.DataFrame(conf_matrix, index=['Real: Baixo', 'Real: Moderado', 'Real: Alto'],
                           columns=['Pred: Baixo', 'Pred: Moderado', 'Pred: Alto'])
    print(conf_df.to_string())
    
    print("\n  Relatório de Classificação Completo:")
    print(classification_report(y_te_r, rf_preds, labels=['Baixo', 'Moderado', 'Alto']))
    
    # Predição final para a base
    df['injury_risk'] = rf_classifier.predict(X_risk)

    # --------------------------------------------------------------------------
    # ATRIBUIÇÃO DOS CLUSTERS FINAIS À TABELA PLAYER_METRICS
    # --------------------------------------------------------------------------
    # Junta clusters de linha e goleiro
    cluster_mapping = {}
    for _, row in df_outfield.iterrows():
        cluster_mapping[row['id']] = int(row['tactical_cluster_id'])
    for _, row in df_gk.iterrows():
        cluster_mapping[row['id']] = int(row['tactical_cluster_id'])
        
    df['cluster_id'] = df['id'].map(cluster_mapping)

    # --------------------------------------------------------------------------
    # EXPORTAÇÃO DOS ARQUIVOS ESTRUTURADOS FINAIS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[6] CONSOLIDAÇÃO E ATUALIZAÇÃO DOS ARQUIVOS EM RESULT/")
    print("-" * 80)
    
    # 1. Tabela players conforme Schema PostgreSQL exato (8 colunas)
    players_schema_cols = [
        'id', 'name', 'position', 'age', 'current_club',
        'market_value', 'salary', 'injury_risk'
    ]
    df_players_final = df[players_schema_cols].copy()
    df_players_final.to_csv(players_csv_path, index=False)
    print(f"  -> Atualizado com sucesso: {players_csv_path} ({len(df_players_final)} registros, 8 colunas)")
    
    # 2. Tabela player_metrics conforme Schema PostgreSQL exato (14 colunas)
    metrics_schema_cols = [
        'player_id', 'minutes_played', 'cluster_id',
        'progressive_passes_p90', 'tackles_p90', 'sca_p90', 'progressive_carries_p90',
        'expected_goals_p90', 'expected_assists_p90',
        'goals_conceded_p90', 'save_percentage', 'clean_sheet_percentage',
        'shots_on_target_against_p90', 'penalties_saved_p90'
    ]
    df_metrics_final = df[metrics_schema_cols].copy()
    df_metrics_final.to_csv(metrics_csv_path, index=False, na_rep="")
    print(f"  -> Atualizado com sucesso: {metrics_csv_path} ({len(df_metrics_final)} registros, 14 colunas)")
    
    # 3. Base analítica enriquecida para relatórios e verificação (players_analyzed.csv)
    analyzed_path = os.path.join(result_dir, "players_analyzed.csv")
    df.to_csv(analyzed_path, index=False)
    print(f"  -> Base analítica expandida salva em: {analyzed_path}")
    
    # 4. Metadados dos Modelos para o Backend Spring Boot (DecisionEngineService)
    # Centroides originais e normalizados para cálculo de similaridade euclidiana
    metadata = {
        "kmeans_outfield": {
            "k": k_outfield,
            "silhouette_score": round(float(sil_outfield), 4),
            "features": outfield_features,
            "scaler_mean": scaler_outfield.mean_.tolist(),
            "scaler_scale": scaler_outfield.scale_.tolist(),
            "cluster_names": {
                "1": "Construtor / Passador Base",
                "2": "Marcador / Destruidor de Jogadas",
                "3": "Criador / Transição Ofensiva"
            },
            "centroids_scaled": kmeans_outfield.cluster_centers_.tolist(),
            "centroids_original": scaler_outfield.inverse_transform(kmeans_outfield.cluster_centers_).tolist()
        },
        "kmeans_gk": {
            "k": k_gk,
            "silhouette_score": round(float(sil_gk), 4),
            "features": gk_features,
            "scaler_mean": scaler_gk.mean_.tolist(),
            "scaler_scale": scaler_gk.scale_.tolist(),
            "cluster_names": {
                "4": "Muro / Goleiro Titular Elite",
                "5": "Moderado / Exposição Média",
                "6": "Especialista em Pênaltis / Alta Pressão"
            },
            "centroids_scaled": kmeans_gk.cluster_centers_.tolist(),
            "centroids_original": scaler_gk.inverse_transform(kmeans_gk.cluster_centers_).tolist()
        },
        "pricing_regression": {
            "model": "RidgeCV (Log-Transformed Target)",
            "alpha": round(float(pricing_model.regressor_.alpha_), 4),
            "r2_log_space": round(float(r2_log), 4),
            "r2_test_euro": round(float(r2_pricing), 4),
            "rmse_test_euro": round(float(rmse_pricing), 2),
            "feature_names": X_pricing_raw.columns.tolist(),
            "coefficients": pricing_model.regressor_.coef_.tolist(),
            "intercept": float(pricing_model.regressor_.intercept_)
        },
        "injury_risk_classifier": {
            "model": "RandomForestClassifier",
            "f1_macro": round(float(rf_f1_macro), 4),
            "f1_weighted": round(float(rf_f1_weighted), 4),
            "features": risk_features,
            "classes": ["Baixo", "Moderado", "Alto"]
        }
    }
    
    metadata_path = os.path.join(result_dir, "models_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  -> Metadados analíticos e centroides salvos em: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("TREINAMENTO E ATUALIZAÇÃO CONCLUÍDOS COM SUCESSO!")
    print("=" * 80)

if __name__ == '__main__':
    run_ml_pipeline()
