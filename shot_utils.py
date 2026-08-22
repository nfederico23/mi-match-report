"""
shot_utils.py
-------------
Lógica de normalización del shotmap de 365Scores, reconstruida a partir del
log de diagnóstico que compartiste (no del código original, que no tenemos).

Si algún nombre de columna no coincide con lo que te devuelve LanusStats en tu
versión, ajustá los diccionarios de mapeo de más abajo — están separados
justamente para que sea fácil tocarlos sin romper el resto.
"""

import pandas as pd


# Mapeo de shot_outcome (español, como lo devuelve 365Scores) -> shotType corto
OUTCOME_TO_SHOTTYPE = {
    "Fallado": "miss",
    "Bloqueado": "block",
    "Atajado": "save",
    "Poste": "post",
    "Gol": "goal",
}

# Mapeo de shotType corto -> nombre estilo Opta (como en el PDF de referencia)
SHOTTYPE_TO_OPTA = {
    "miss": "MissedShots",
    "block": "BlockedShot",
    "save": "SavedShot",
    "post": "ShotOnPost",
    "goal": "Goal",
}


def _resolve_team_column(df, cols_map, team_col):
    """
    Elige la columna que identifica al equipo. Si el usuario no especifica una,
    probamos varios nombres candidatos y nos quedamos con el primero que tenga
    2 (o pocos) valores distintos — porque un partido siempre tiene 2 equipos.

    Esto evita el bug de confiar ciegamente en el nombre de columna: en la
    práctica vimos que "side" en el shotmap real de 365Scores NO es home/away,
    tiene un valor distinto por cada tiro (parece ser algo tipo minuto/tiempo).
    "competitorNum" sí resultó ser 1/2 (el equipo real).
    """
    candidates = [team_col] if team_col else ["competitorNum", "teamName", "team", "team_name", "side"]
    present = [cols_map[c.lower()] for c in candidates if c and c.lower() in cols_map]

    if not present:
        raise KeyError(
            f"No encontré ninguna columna de equipo entre {candidates} en el shotmap. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Preferimos la primera columna presente cuya cantidad de valores distintos
    # sea razonable para "equipo" (1, 2, o 3 por las dudas de datos incompletos).
    for col in present:
        if df[col].nunique(dropna=True) <= 3:
            return col

    # Si ninguna cumple, devolvemos la primera igual — mejor eso que romper,
    # aunque el resultado probablemente no agrupe bien por equipo.
    return present[0]


def normalize_shotmap(shots_df, outcome_col="shot_outcome", team_col=None):
    """
    Toma el shotmap crudo de 365Scores y devuelve:
      - shots_df con columnas shotType / type agregadas
      - un resumen agrupado por equipo + tipo con cantidad, xG, xGOT

    Replica la secuencia que se ve en tu log:
        shot_outcome (es) -> shotType (corto) -> type (estilo Opta) -> groupby

    team_col=None (default): se resuelve automáticamente por cardinalidad —
    ver _resolve_team_column().
    """
    df = shots_df.copy()
    cols_map = {c.lower(): c for c in df.columns}

    resolved_outcome = cols_map.get(outcome_col.lower())
    if resolved_outcome is None:
        raise KeyError(
            f"No encontré la columna '{outcome_col}' en el shotmap. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    outcome_col = resolved_outcome
    team_col = _resolve_team_column(df, cols_map, team_col)

    df["shotType"] = df[outcome_col].map(OUTCOME_TO_SHOTTYPE)
    df["type"] = df["shotType"].map(SHOTTYPE_TO_OPTA)

    agg = {}
    if "xg" in [c.lower() for c in df.columns]:
        xg_col = next(c for c in df.columns if c.lower() == "xg")
        agg[xg_col] = "sum"
    if "xgot" in [c.lower() for c in df.columns]:
        xgot_col = next(c for c in df.columns if c.lower() == "xgot")
        agg[xgot_col] = "sum"

    grouped = (
        df.groupby([team_col, outcome_col, "shotType", "type"], dropna=False)
        .agg(cantidad=("shotType", "size"), **({k: (v, v) for k, v in {}} or {}))
        .reset_index()
    )
    # Sumamos xG/xGOT aparte para no pelear con la sintaxis de agg mixta
    if agg:
        extra = df.groupby([team_col, outcome_col, "shotType", "type"], dropna=False).agg(agg).reset_index()
        grouped = grouped.merge(extra, on=[team_col, outcome_col, "shotType", "type"], how="left")

    return df, grouped


def team_shot_summary(shots_df, team_col=None, xg_col="xg", xgot_col="xgot"):
    """Totales por equipo: tiros, tiros al arco, xG, xGOT — como el panel central del reporte."""
    df = shots_df.copy()
    cols = {c.lower(): c for c in df.columns}
    xg_col = cols.get(xg_col.lower())
    xgot_col = cols.get(xgot_col.lower())
    shottype_col = "shotType" if "shotType" in df.columns else None
    team_col = _resolve_team_column(df, cols, team_col)

    summary = {}
    for team, sub in df.groupby(team_col):
        total_shots = len(sub)
        on_target = 0
        if shottype_col:
            on_target = sub[shottype_col].isin(["save", "goal"]).sum()
        summary[team] = {
            "tiros": total_shots,
            "al_arco": int(on_target),
            "xg": round(sub[xg_col].sum(), 2) if xg_col else None,
            "xgot": round(sub[xgot_col].sum(), 2) if xgot_col else None,
        }
    return summary


def pivot_long_general_stats(general_stats_df, name_col="name", team_col="competitorId",
                              value_col="value", major_only=True):
    """
    365Scores devuelve get_match_general_stats en formato "largo": una fila por
    cada (estadística, equipo), con columnas id/name/competitorId/isMajor/value.
    Esto lo pivotea a {nombre_stat: (valor_equipo_1, valor_equipo_2)}, limpiando
    valores con "%" a número.

    major_only=True: solo usa las filas marcadas isMajor (las más relevantes,
    como en el panel "Match Stats" del PDF de referencia) si esa columna existe.
    """
    df = general_stats_df.copy()
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get(name_col.lower(), df.columns[0])
    team_col = cols.get(team_col.lower())
    value_col = cols.get(value_col.lower(), df.columns[-1])
    major_col = cols.get("ismajor")

    if team_col is None:
        raise KeyError(f"No encontré columna de equipo tipo 'competitorId'. Columnas: {list(df.columns)}")

    if major_only and major_col and df[major_col].notna().any():
        df = df[df[major_col].fillna(False).astype(bool)]

    def clean_value(v):
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", ".").strip()
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    df["_value_clean"] = df[value_col].map(clean_value)

    team_ids = sorted(df[team_col].dropna().unique(), key=str)
    stats = {}
    for stat_name, sub in df.groupby(name_col):
        vals = []
        for tid in team_ids[:2]:
            row = sub[sub[team_col] == tid]
            vals.append(row["_value_clean"].iloc[0] if not row.empty else None)
        if len(vals) == 2:
            stats[stat_name] = tuple(vals)

    return stats


def convert_goalmouth_to_plot(df, y_col="goalMouthY", z_col="goalMouthZ"):
    """
    Convierte las coordenadas crudas de arco (goalMouthY/Z, escala 0-100 aprox.
    tipo 365Scores) a coordenadas de ploteo sobre un arco dibujado en mplsoccer
    (ancho ~7.32m / alto ~2.44m, expresado en el sistema opta 0-100 horizontal).

    OJO: esta conversión está reconstruida a partir de los valores que se ven
    en tu log (ej. goalMouthY=52.0 -> goalMouthY_plot=35.78), no del código
    original. Si los valores no calzan visualmente al probarlo, es el primer
    lugar para ajustar la fórmula.

    IMPORTANTE: el shotmap real que devuelve get_match_shotmap() de LanusStats
    (365Scores) NO trae columnas goalMouthY/goalMouthZ — esas se calculaban en
    el notebook original con datos extra que no tenemos. Por ahora el panel de
    "Goles y atajadas" en app.py va a mostrar "Sin datos de arco" hasta que
    consigamos de dónde sacar esas coordenadas (o las derivemos de x/y/z, que
    sí están disponibles).
    """
    df = df.copy()
    # Del log: rango observado de goalMouthY ~ [45.9, 53.6] -> goalMouthY_plot ~ [22.4, 86.8]
    # Se ve una transformación lineal tipo: plot = (y - offset) * escala
    # Aproximación conservadora (ajustar si no calza):
    df[f"{y_col}_plot"] = (df[y_col] - 50) * 8.9 + 45
    df[f"{z_col}_metros"] = df[z_col] * 0.061  # aprox: 100 -> 6.1m visual
    df[f"{z_col}_pct_arco"] = df[z_col] / 40.0  # aprox: 100 -> 2.5 (pct del alto de arco)
    df[f"{z_col}_plot"] = df[z_col] * 0.75

    return df
