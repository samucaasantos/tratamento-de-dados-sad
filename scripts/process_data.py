"""
FootDecision SAD - Pipeline de Engenharia e Ciência de Dados
Harmonização, Tratamento, Filtros Estatísticos e Engenharia de Atributos (p90)
Conforme schema PostgreSQL do FootDecision SAD
"""

import os
import sys
import re
import shutil
import unicodedata
import pandas as pd
import numpy as np

try:
    from rapidfuzz import process, fuzz
except ImportError:
    import difflib

    class _FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0

    class _ProcessFallback:
        @staticmethod
        def extractOne(query: str, choices: list, scorer=None):
            if not choices:
                return None
            best_match = None
            best_score = -1.0
            for c in choices:
                score = difflib.SequenceMatcher(None, query, c).ratio() * 100.0
                if score > best_score:
                    best_score = score
                    best_match = c
            return (best_match, best_score) if best_match is not None else None

    fuzz = _FuzzFallback()
    process = _ProcessFallback()

# ==============================================================================
# 1. FUNÇÕES DE NORMALIZAÇÃO E RESOLUÇÃO DE ENTIDADES
# ==============================================================================

def normalize_name(s: str) -> str:
    """
    Normalização robusta de nomes de atletas:
    - Minúsculas (.lower())
    - Remoção de acentos e diacríticos via unicodedata NFKD
    - Remoção de sufixos/prefixos comuns (Jr, Junior, Sr, II, III, etc.)
    - Remoção de pontuações, traços e espaços múltiplos
    """
    if not isinstance(s, str):
        return ""
    # Decomposição de caracteres acentuados
    s = unicodedata.normalize('NFKD', s)
    # Remove marcas diacríticas
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Remove sufixos e prefixos geracionais / comuns
    s = re.sub(r'\b(jr\.?|junior|júnior|sr\.?|senior|ii|iii|iv|filho|neto|sobrinho)\b', '', s)
    # Substitui pontuações e caracteres não alfanuméricos por espaço
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Colapsa múltiplos espaços em um único e remove bordas
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def standardize_position(pos_str: str) -> str:
    """
    Padroniza posição textual para uma das 4 categorias: GK, DF, MF, FW.
    Trata posições compostas (ex: MF,FW -> MF; DF,MF -> DF).
    """
    if not isinstance(pos_str, str):
        return ""
    pos_clean = pos_str.upper().strip()
    primary = pos_clean.split(',')[0].strip()
    if 'GK' in primary:
        return 'GK'
    elif 'DF' in primary:
        return 'DF'
    elif 'MF' in primary:
        return 'MF'
    elif 'FW' in primary:
        return 'FW'
    return primary

# ==============================================================================
# 2. PIPELINE PRINCIPAL DE PROCESSAMENTO
# ==============================================================================

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_dir = os.path.join(base_dir, "scripts")
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    # Caminhos dos arquivos de entrada
    fbref_path = os.path.join(base_dir, "player-stats", "players_data_light-2026_2027.csv")
    wages_path = os.path.join(base_dir, "wages", "dfAll.csv")
    tm_players_path = os.path.join(base_dir, "transfermarket", "players.csv")
    injuries_path = os.path.join(base_dir, "injuries", "full_dataset_thesis - 1.csv")

    print("=" * 80)
    print("FOOTDECISION SAD - PIPELINE DE HARMONIZAÇÃO E PROCESSAMENTO DE DADOS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # PASSO 1: CARGA DAS BASES DE DADOS
    # --------------------------------------------------------------------------
    print("\n[PASSO 1] Carregando bases de dados brutas...")
    df_fbref_raw = pd.read_csv(fbref_path)
    total_fbref_raw = len(df_fbref_raw)
    print(f"  -> FBref (Estatísticas e Desempenho): {total_fbref_raw:,} registros carregados.")

    df_tm_raw = pd.read_csv(tm_players_path)
    total_tm_raw = len(df_tm_raw)
    print(f"  -> Transfermarkt (Jogadores e Valores): {total_tm_raw:,} registros carregados.")

    df_wages_raw = pd.read_csv(wages_path)
    total_wages_raw = len(df_wages_raw)
    print(f"  -> Capology/Undervalued (Salários e Financeiro): {total_wages_raw:,} registros carregados.")

    df_injuries_raw = pd.read_csv(injuries_path)
    total_injuries_raw = len(df_injuries_raw)
    print(f"  -> Transfermarkt (Histórico de Lesões): {total_injuries_raw:,} registros carregados.")

    # --------------------------------------------------------------------------
    # PASSO 2: FILTRO DE MINUTAGEM E REGRAS ESTRITAS DE CORTE
    # --------------------------------------------------------------------------
    print("\n[PASSO 2] Aplicando filtro de minutagem estrita (450 <= Min <= 900)...")
    mask_min = (df_fbref_raw['Min'] >= 450) & (df_fbref_raw['Min'] <= 900)
    df_fbref_filtered = df_fbref_raw[mask_min].copy()
    discarded_by_minutes = total_fbref_raw - len(df_fbref_filtered)
    print(f"  -> Atletas descartados por minutagem (< 450 ou > 900 min): {discarded_by_minutes:,}")
    print(f"  -> Atletas elegíveis para harmonização: {len(df_fbref_filtered):,}")

    # Normaliza nomes do FBref
    df_fbref_filtered['norm_name'] = df_fbref_filtered['Player'].apply(normalize_name)

    # --------------------------------------------------------------------------
    # PASSO 3: PREPARAÇÃO DAS BASES AUXILIARES (TM, WAGES, INJURIES)
    # --------------------------------------------------------------------------
    print("\n[PASSO 3] Preparando bases auxiliares e normalizando identidades...")
    
    # Transfermarkt: normalização e desambiguação
    df_tm = df_tm_raw.copy()
    df_tm['norm_name'] = df_tm['name'].apply(normalize_name)
    df_tm['full_norm'] = (df_tm['first_name'].fillna('') + ' ' + df_tm['last_name'].fillna('')).apply(normalize_name)
    df_tm_valid = df_tm.dropna(subset=['market_value_in_eur']).copy()
    print(f"  -> Atletas no Transfermarkt com market_value não-nulo: {len(df_tm_valid):,}")

    # Wages (Capology): normalização e cálculo de salário mensal
    df_wages = df_wages_raw.copy()
    df_wages['norm_name'] = df_wages['Player'].apply(normalize_name)
    # Salário anual em euros dividido por 12 = salário mensal em euros
    df_wages['monthly_salary'] = (df_wages['Annual EUR'] / 12.0).round(2)
    df_wages_valid = df_wages.dropna(subset=['monthly_salary']).copy()
    print(f"  -> Atletas em Wages com salário válido: {len(df_wages_valid):,}")

    # Injuries: filtro temporal de 2023 em diante e agregações
    print("\n  -> Processando histórico médico de lesões (>= 2023)...")
    # Parse datas
    inj_dates = pd.to_datetime(df_injuries_raw['injury_from_parsed'], errors='coerce')
    # Lesões a partir de 2023
    mask_recent_inj = (inj_dates.dt.year >= 2023) | (df_injuries_raw['Season'].isin(['23/24', '24/25']))
    df_inj_recent = df_injuries_raw[mask_recent_inj].copy()
    
    # Extrai dias numéricos
    days_numeric = df_inj_recent['Days'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
    df_inj_recent['days_clean'] = days_numeric
    df_inj_recent['games_missed_clean'] = pd.to_numeric(df_inj_recent['Games missed'], errors='coerce').fillna(0).astype(int)
    df_inj_recent['norm_name'] = df_inj_recent['player_name'].apply(normalize_name)

    # Agregação por atleta
    injury_agg = df_inj_recent.groupby('norm_name').agg(
        total_days_injured=('days_clean', 'sum'),
        total_games_missed=('games_missed_clean', 'sum'),
        injury_count=('days_clean', 'count')
    ).reset_index()
    print(f"  -> Atletas com registros de lesões recentes (>= 2023): {len(injury_agg):,}")

    # --------------------------------------------------------------------------
    # PASSO 4: ENTITY RESOLUTION E HARMONIZAÇÃO DE NOMES (FUZZY MATCHING >= 92%)
    # --------------------------------------------------------------------------
    print("\n[PASSO 4] Executando resolução de entidades e junção estrita...")

    tm_name_set = set(df_tm_valid['norm_name'].dropna())
    tm_name_list = list(tm_name_set)
    wages_name_set = set(df_wages_valid['norm_name'].dropna())
    wages_name_list = list(wages_name_set)

    matched_records = []
    discarded_no_data = 0

    for idx, row in df_fbref_filtered.iterrows():
        fb_norm = row['norm_name']
        fb_player = row['Player']
        fb_squad = str(row['Squad']).lower().strip()
        fb_pos = standardize_position(row['Pos'])
        fb_age = row['Age']
        fb_born = row['Born']

        # ----------------------------------------------------------------------
        # 1. Correspondência com Transfermarkt (Valor de Mercado)
        # ----------------------------------------------------------------------
        matched_tm_row = None
        if fb_norm in tm_name_set:
            cands = df_tm_valid[df_tm_valid['norm_name'] == fb_norm]
            if len(cands) == 1:
                matched_tm_row = cands.iloc[0]
            else:
                # Desambiguação por clube ou ano de nascimento
                club_cands = cands[cands['current_club_name'].astype(str).str.lower().apply(
                    lambda c: c in fb_squad or fb_squad in c
                )]
                if len(club_cands) >= 1:
                    matched_tm_row = club_cands.iloc[0]
                elif pd.notna(fb_born):
                    dob_cands = cands[cands['date_of_birth'].astype(str).str.startswith(str(int(fb_born)))]
                    if len(dob_cands) >= 1:
                        matched_tm_row = dob_cands.iloc[0]
                    else:
                        matched_tm_row = cands.iloc[0]
                else:
                    matched_tm_row = cands.iloc[0]
        else:
            # Fuzzy match com threshold >= 92%
            best_match = process.extractOne(fb_norm, tm_name_list, scorer=fuzz.ratio)
            if best_match and best_match[1] >= 92.0:
                cands = df_tm_valid[df_tm_valid['norm_name'] == best_match[0]]
                if len(cands) >= 1:
                    matched_tm_row = cands.iloc[0]

        # ----------------------------------------------------------------------
        # 2. Correspondência com Capology/Wages (Salário Mensal)
        # ----------------------------------------------------------------------
        matched_wage_row = None
        if fb_norm in wages_name_set:
            cands = df_wages_valid[df_wages_valid['norm_name'] == fb_norm]
            if len(cands) == 1:
                matched_wage_row = cands.iloc[0]
            else:
                # Desambiguação por clube ou idade
                club_cands = cands[cands['Squad_x'].astype(str).str.lower().apply(
                    lambda c: c in fb_squad or fb_squad in c
                )]
                if len(club_cands) >= 1:
                    matched_wage_row = club_cands.iloc[0]
                else:
                    matched_wage_row = cands.iloc[0]
        else:
            # Fuzzy match com threshold >= 92%
            best_match = process.extractOne(fb_norm, wages_name_list, scorer=fuzz.ratio)
            if best_match and best_match[1] >= 92.0:
                cands = df_wages_valid[df_wages_valid['norm_name'] == best_match[0]]
                if len(cands) >= 1:
                    matched_wage_row = cands.iloc[0]

        # ----------------------------------------------------------------------
        # 3. Verificação Estrita de Integridade (Descarte por Incompletude)
        # ----------------------------------------------------------------------
        if matched_tm_row is None or matched_wage_row is None:
            discarded_no_data += 1
            continue

        market_val = matched_tm_row['market_value_in_eur']
        salary_val = matched_wage_row['monthly_salary']

        # Se algum dos atributos mandatórios for nulo, descartar sumariamente
        if pd.isna(market_val) or pd.isna(salary_val) or pd.isna(fb_age) or not fb_pos:
            discarded_no_data += 1
            continue

        # ----------------------------------------------------------------------
        # 4. Métricas de Lesão Agregadas
        # ----------------------------------------------------------------------
        inj_match = injury_agg[injury_agg['norm_name'] == fb_norm]
        if len(inj_match) > 0:
            tot_days = int(inj_match.iloc[0]['total_days_injured'])
            tot_games = int(inj_match.iloc[0]['total_games_missed'])
            tot_inj_count = int(inj_match.iloc[0]['injury_count'])
        else:
            tot_days = 0
            tot_games = 0
            tot_inj_count = 0

        # Coleta registro consolidado
        matched_records.append({
            'fbref_row': row,
            'tm_row': matched_tm_row,
            'wage_row': matched_wage_row,
            'norm_name': fb_norm,
            'name': fb_player,
            'position': fb_pos,
            'age': int(fb_age),
            'current_club': row['Squad'],
            'market_value': round(float(market_val), 2),
            'salary': round(float(salary_val), 2),
            'total_days_injured': tot_days,
            'total_games_missed': tot_games,
            'injury_count': tot_inj_count,
            'is_goalkeeper': (fb_pos == 'GK')
        })

    print(f"  -> Atletas descartados por dados cadastrais/financeiros incompletos: {discarded_no_data:,}")
    print(f"  -> Atletas validados e consolidados: {len(matched_records):,}")

    # --------------------------------------------------------------------------
    # PASSO 5: ENGENHARIA DE ATRIBUTOS (MÉTRICAS POR 90 MINUTOS p90)
    # --------------------------------------------------------------------------
    print("\n[PASSO 5] Calculando métricas p90 bifurcadas por posição (Linha vs Goleiros)...")

    players_list = []
    metrics_list = []

    for idx, rec in enumerate(matched_records, start=1):
        p_id = idx
        fb_row = rec['fbref_row']
        wage_row = rec['wage_row']
        pos = rec['position']
        min_played = int(fb_row['Min'])
        n90 = float(fb_row['90s']) if float(fb_row['90s']) > 0 else min_played / 90.0

        # Tabela players
        players_list.append({
            'id': p_id,
            'name': rec['name'],
            'position': pos,
            'age': rec['age'],
            'current_club': rec['current_club'],
            'market_value': f"{rec['market_value']:.2f}",
            'salary': f"{rec['salary']:.2f}",
            'total_days_injured': rec['total_days_injured'],
            'total_games_missed': rec['total_games_missed'],
            'injury_count': rec['injury_count'],
            'is_goalkeeper': rec['is_goalkeeper']
        })

        # Tabela player_metrics
        # Bifurcação estrita Linha vs Goleiro
        if pos != 'GK':
            # --- JOGADORES DE LINHA ---
            # tackles_p90 = TklW / 90s
            tklw = float(fb_row['TklW']) if pd.notna(fb_row['TklW']) else 0.0
            tackles_p90 = round(tklw / n90, 4)

            # sca_p90 = SCA / 90s (ou (SoT + Ast) / 90s como proxy explícito)
            sot = float(fb_row['SoT']) if pd.notna(fb_row['SoT']) else 0.0
            ast = float(fb_row['Ast']) if pd.notna(fb_row['Ast']) else 0.0
            sca_p90 = round((sot + ast) / n90, 4)

            # expected_goals_p90 = xG / 90s (ou Gls / 90s)
            if 'xG' in wage_row and pd.notna(wage_row['xG']) and float(wage_row['xG']) > 0:
                xg_val = float(wage_row['xG'])
                # normalizado por 90s da época ou da minutagem
                xg_p90 = round(xg_val / n90, 4)
            else:
                gls = float(fb_row['Gls']) if pd.notna(fb_row['Gls']) else 0.0
                xg_p90 = round(gls / n90, 4)

            # expected_assists_p90 = xAG / 90s (ou Ast / 90s)
            if 'xAG' in wage_row and pd.notna(wage_row['xAG']) and float(wage_row['xAG']) > 0:
                xag_val = float(wage_row['xAG'])
                xag_p90 = round(xag_val / n90, 4)
            else:
                xag_p90 = round(ast / n90, 4)

            # progressive_passes_p90 = PrgP / 90s (ou proxy direto correlacionado)
            if 'PrgP' in wage_row and pd.notna(wage_row['PrgP']) and float(wage_row['PrgP']) > 0:
                prgp_val = float(wage_row['PrgP'])
                prgp_p90 = round(prgp_val / n90, 4)
            else:
                crs = float(fb_row['Crs']) if pd.notna(fb_row['Crs']) else 0.0
                prgp_p90 = round((crs + ast) / n90, 4)

            # progressive_carries_p90 = PrgC / 90s (ou conduções terço final / proxy)
            if 'PrgC' in wage_row and pd.notna(wage_row['PrgC']) and float(wage_row['PrgC']) > 0:
                prgc_val = float(wage_row['PrgC'])
                prgc_p90 = round(prgc_val / n90, 4)
            else:
                sh = float(fb_row['Sh']) if pd.notna(fb_row['Sh']) else 0.0
                fld = float(fb_row['Fld']) if pd.notna(fb_row['Fld']) else 0.0
                prgc_p90 = round((sh + fld) / n90, 4)

            # Colunas de goleiro são gravadas como NULL
            metrics_list.append({
                'player_id': p_id,
                'minutes_played': min_played,
                'cluster_id': 0,
                'progressive_passes_p90': prgp_p90,
                'tackles_p90': tackles_p90,
                'sca_p90': sca_p90,
                'progressive_carries_p90': prgc_p90,
                'expected_goals_p90': xg_p90,
                'expected_assists_p90': xag_p90,
                'goals_conceded_p90': np.nan,
                'save_percentage': np.nan,
                'clean_sheet_percentage': np.nan,
                'shots_on_target_against_p90': np.nan,
                'penalties_saved_p90': np.nan
            })
        else:
            # --- GOLEIROS (Pos == 'GK') ---
            # goals_conceded_p90 = GA / 90s (ou coluna direta GA90)
            if pd.notna(fb_row['GA90']):
                ga_p90 = round(float(fb_row['GA90']), 4)
            elif pd.notna(fb_row['GA']):
                ga_p90 = round(float(fb_row['GA']) / n90, 4)
            else:
                ga_p90 = 0.0

            # save_percentage = Save%
            if pd.notna(fb_row['Save%']):
                save_pct = round(float(fb_row['Save%']), 2)
            elif pd.notna(fb_row['Saves']) and pd.notna(fb_row['SoTA']) and float(fb_row['SoTA']) > 0:
                save_pct = round((float(fb_row['Saves']) / float(fb_row['SoTA'])) * 100.0, 2)
            else:
                save_pct = 0.0

            # clean_sheet_percentage = CS% (jogos sem sofrer gols / total partidas)
            if pd.notna(fb_row['CS%']):
                cs_pct = round(float(fb_row['CS%']), 2)
            elif pd.notna(fb_row['CS']) and pd.notna(fb_row['MP']) and float(fb_row['MP']) > 0:
                cs_pct = round((float(fb_row['CS']) / float(fb_row['MP'])) * 100.0, 2)
            else:
                cs_pct = 0.0

            # shots_on_target_against_p90 = SoTA / 90s
            sota = float(fb_row['SoTA']) if pd.notna(fb_row['SoTA']) else 0.0
            sota_p90 = round(sota / n90, 4)

            # penalties_saved_p90 = PKsv / 90s
            pksv = float(fb_row['PKsv']) if pd.notna(fb_row['PKsv']) else 0.0
            pksv_p90 = round(pksv / n90, 4)

            # Colunas de linha são gravadas como NULL
            metrics_list.append({
                'player_id': p_id,
                'minutes_played': min_played,
                'cluster_id': 0,
                'progressive_passes_p90': np.nan,
                'tackles_p90': np.nan,
                'sca_p90': np.nan,
                'progressive_carries_p90': np.nan,
                'expected_goals_p90': np.nan,
                'expected_assists_p90': np.nan,
                'goals_conceded_p90': ga_p90,
                'save_percentage': save_pct,
                'clean_sheet_percentage': cs_pct,
                'shots_on_target_against_p90': sota_p90,
                'penalties_saved_p90': pksv_p90
            })

    df_players_out = pd.DataFrame(players_list)
    df_metrics_out = pd.DataFrame(metrics_list)

    # --------------------------------------------------------------------------
    # PASSO 6: EXPORTAÇÃO DOS ARQUIVOS ESTRUTURADOS
    # --------------------------------------------------------------------------
    print("\n[PASSO 6] Exportando arquivos CSV estruturados...")

    # Salva no diretório result/
    players_result_path = os.path.join(result_dir, "players_cleaned.csv")
    metrics_result_path = os.path.join(result_dir, "player_metrics_cleaned.csv")

    df_players_out.to_csv(players_result_path, index=False, na_rep="")
    df_metrics_out.to_csv(metrics_result_path, index=False, na_rep="")
    print(f"  -> Salvo com sucesso: {players_result_path}")
    print(f"  -> Salvo com sucesso: {metrics_result_path}")

    # Salva exclusivamente no diretório result/

    # --------------------------------------------------------------------------
    # PASSO 7: RELATÓRIO DE EXECUÇÃO E AUDITORIA DE INTEGRIDADE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RELATÓRIO DE EXECUÇÃO E AUDITORIA DE INTEGRIDADE (FOOTDECISION SAD)")
    print("=" * 80)

    print("\n1. TOTAL DE REGISTROS LIDOS NAS BASES DE ENTRADA:")
    print(f"   - Base FBref (Estatísticas e Desempenho) : {total_fbref_raw:>8,} registros")
    print(f"   - Base Transfermarkt (Cadastral / Valores): {total_tm_raw:>8,} registros")
    print(f"   - Base Capology (Salários e Financeiro)  : {total_wages_raw:>8,} registros")
    print(f"   - Base Transfermarkt Injuries (Lesões)   : {total_injuries_raw:>8,} registros")

    print("\n2. ESTATÍSTICAS DE FILTRAGEM E POLÍTICA DE DESCARTE:")
    print(f"   - Atletas descartados por corte de minutagem (< 450 ou > 900 min): {discarded_by_minutes:>6,}")
    print(f"   - Atletas descartados por ausência de dados cadastrais/financeiros: {discarded_no_data:>6,}")

    total_final = len(df_players_out)
    gk_count = (df_players_out['position'] == 'GK').sum()
    outfield_count = (df_players_out['position'] != 'GK').sum()

    print("\n3. VOLUME FINAL DE ATLETAS CONSOLIDADOS:")
    print(f"   - Total de atletas consolidados: {total_final:>6,}")
    print(f"   - Jogadores de Linha (Outfield): {outfield_count:>6,} ({outfield_count/total_final*100:.1f}%)")
    print(f"   - Goleiros (Goalkeepers)       : {gk_count:>6,} ({gk_count/total_final*100:.1f}%)")

    print("\n4. VERIFICAÇÃO DE INTEGRIDADE (CHECAGEM DE NULOS EM COLUNAS OBRIGATÓRIAS):")
    mandatory_players_cols = ['id', 'name', 'position', 'age', 'current_club', 'market_value', 'salary']
    nulls_players = df_players_out[mandatory_players_cols].isnull().sum()
    print("   [Tabela players]")
    for col, count in nulls_players.items():
        print(f"     - Coluna '{col}': {count} nulos (Status: {'OK - 100% Íntegro' if count == 0 else 'FALHA'})")

    mandatory_metrics_cols = ['player_id', 'minutes_played', 'cluster_id']
    nulls_metrics = df_metrics_out[mandatory_metrics_cols].isnull().sum()
    print("   [Tabela player_metrics]")
    for col, count in nulls_metrics.items():
        print(f"     - Coluna '{col}': {count} nulos (Status: {'OK - 100% Íntegro' if count == 0 else 'FALHA'})")

    print("\n5. EXIBIÇÃO DAS 5 PRIMEIRAS LINHAS DE CADA TABELA GERADA:")
    print("\n--- [players_cleaned.csv] (Primeiras 5 linhas) ---")
    print(df_players_out.head(5).to_string(index=False))

    print("\n--- [player_metrics_cleaned.csv] (Primeiras 5 linhas) ---")
    print(df_metrics_out.head(5).to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    main()
