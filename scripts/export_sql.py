"""
FootDecision SAD - Exportação da Base Consolidada (SQL Generator)
Gera o script data.sql para popular diretamente as tabelas PostgreSQL do Neon/Render:
- players (dados cadastrais, contratuais e rótulos de risco)
- player_metrics (minutagem, clusters K-Means e atributos estatísticos p90)
"""

import os
import pandas as pd
import numpy as np

def escape_sql_string(val):
    if val is None or pd.isna(val):
        return "NULL"
    s = str(val).strip()
    s = s.replace("'", "''")
    return f"'{s}'"

def format_sql_number(val, decimal_places=2):
    if val is None or pd.isna(val) or val == "":
        return "NULL"
    try:
        f = float(val)
        return f"{f:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "NULL"

def format_sql_int(val):
    if val is None or pd.isna(val) or val == "":
        return "NULL"
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return "NULL"

def generate_data_sql():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result_dir = os.path.join(base_dir, "result")
    
    players_csv = os.path.join(result_dir, "players_cleaned.csv")
    metrics_csv = os.path.join(result_dir, "player_metrics_cleaned.csv")
    output_sql = os.path.join(result_dir, "data.sql")
    
    print("=" * 80)
    print("FOOTDECISION SAD - GERADOR DE SCRIPT SQL (data.sql)")
    print("=" * 80)
    
    print(f"\n[1] Lendo dados consolidados...")
    df_players = pd.read_csv(players_csv)
    df_metrics = pd.read_csv(metrics_csv)
    
    print(f"  -> players_cleaned.csv: {len(df_players)} linhas")
    print(f"  -> player_metrics_cleaned.csv: {len(df_metrics)} linhas")
    
    lines = []
    lines.append("-- =============================================================================")
    lines.append("-- FOOTDECISION SAD - CARGA INICIAL DE DADOS (data.sql)")
    lines.append("-- Populando tabelas 'players' e 'player_metrics' com clusters e rótulos inferidos")
    lines.append("-- =============================================================================")
    lines.append("")
    lines.append("-- Limpeza prévia ANSI SQL (compatível com PostgreSQL e H2)")
    lines.append("DELETE FROM player_metrics;")
    lines.append("DELETE FROM players;")
    lines.append("")
    
    # --------------------------------------------------------------------------
    # 1. INSERT INTO players
    # --------------------------------------------------------------------------
    lines.append("-- -----------------------------------------------------------------------------")
    lines.append(f"-- 1. INSERTS NA TABELA players ({len(df_players)} registros)")
    lines.append("-- Colunas: id, name, position, age, current_club, market_value, fair_market_value, salary, injury_risk")
    lines.append("-- -----------------------------------------------------------------------------")
    lines.append("INSERT INTO players (id, name, position, age, current_club, market_value, fair_market_value, salary, injury_risk) VALUES")
    
    player_values = []
    for _, row in df_players.iterrows():
        p_id = format_sql_int(row['id'])
        name = escape_sql_string(row['name'])
        pos = escape_sql_string(row['position'])
        age = format_sql_int(row['age'])
        club = escape_sql_string(row['current_club'])
        m_val = format_sql_number(row['market_value'], 2)
        fm_val = format_sql_number(row['fair_market_value'], 2)
        salary = format_sql_number(row['salary'], 2)
        risk = escape_sql_string(row['injury_risk'])
        
        player_values.append(f"  ({p_id}, {name}, {pos}, {age}, {club}, {m_val}, {fm_val}, {salary}, {risk})")
        
    lines.append(",\n".join(player_values) + ";")
    lines.append("")
    lines.append("-- Sincronização da sequence do id da tabela players (descomente se usar PostgreSQL puro)")
    lines.append("-- SELECT setval('players_id_seq', (SELECT MAX(id) FROM players));")
    lines.append("")
    
    # --------------------------------------------------------------------------
    # 2. INSERT INTO player_metrics
    # --------------------------------------------------------------------------
    lines.append("-- -----------------------------------------------------------------------------")
    lines.append(f"-- 2. INSERTS NA TABELA player_metrics ({len(df_metrics)} registros)")
    lines.append("-- Colunas: player_id, minutes_played, cluster_id,")
    lines.append("--          progressive_passes_p90, tackles_p90, sca_p90, progressive_carries_p90,")
    lines.append("--          expected_goals_p90, expected_assists_p90,")
    lines.append("--          goals_conceded_p90, save_percentage, clean_sheet_percentage,")
    lines.append("--          shots_on_target_against_p90, penalties_saved_p90")
    lines.append("-- -----------------------------------------------------------------------------")
    lines.append("INSERT INTO player_metrics (")
    lines.append("  player_id, minutes_played, cluster_id,")
    lines.append("  progressive_passes_p90, tackles_p90, sca_p90, progressive_carries_p90,")
    lines.append("  expected_goals_p90, expected_assists_p90,")
    lines.append("  goals_conceded_p90, save_percentage, clean_sheet_percentage,")
    lines.append("  shots_on_target_against_p90, penalties_saved_p90")
    lines.append(") VALUES")
    
    metric_values = []
    for _, row in df_metrics.iterrows():
        p_id = format_sql_int(row['player_id'])
        min_p = format_sql_int(row['minutes_played'])
        c_id = format_sql_int(row['cluster_id'])
        
        prg_p = format_sql_number(row['progressive_passes_p90'], 2)
        tkl = format_sql_number(row['tackles_p90'], 2)
        sca = format_sql_number(row['sca_p90'], 2)
        prg_c = format_sql_number(row['progressive_carries_p90'], 2)
        xg = format_sql_number(row['expected_goals_p90'], 2)
        xa = format_sql_number(row['expected_assists_p90'], 2)
        
        ga = format_sql_number(row['goals_conceded_p90'], 2)
        save_pct = format_sql_number(row['save_percentage'], 2)
        cs_pct = format_sql_number(row['clean_sheet_percentage'], 2)
        sota = format_sql_number(row['shots_on_target_against_p90'], 2)
        pksv = format_sql_number(row['penalties_saved_p90'], 2)
        
        metric_values.append(
            f"  ({p_id}, {min_p}, {c_id}, {prg_p}, {tkl}, {sca}, {prg_c}, {xg}, {xa}, {ga}, {save_pct}, {cs_pct}, {sota}, {pksv})"
        )
        
    lines.append(",\n".join(metric_values) + ";")
    lines.append("")
    
    # Grava o arquivo data.sql
    with open(output_sql, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    file_size_kb = os.path.getsize(output_sql) / 1024
    print(f"\n[2] Script gerado com sucesso!")
    print(f"  -> Destino: {output_sql}")
    print(f"  -> Tamanho: {file_size_kb:.2f} KB")
    print(f"  -> Total de linhas SQL: {len(lines)}")
    print("=" * 80)

if __name__ == '__main__':
    generate_data_sql()
