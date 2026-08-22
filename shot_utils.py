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


def normalize_shotmap(shots_df, outcome_col="shot_outcome", team_col="teamName"):
    """
    Toma el shotmap crudo de 365Scores y devuelve:
      - shots_df con columnas shotType / type agregadas
      - un resumen agrupado por equipo + tipo con cantidad, xG, xGOT

    Replica la secuencia que se ve en tu log:
        shot_outcome (es) -> shotType (corto) -> type (estilo Opta) -> groupby
    """
    df = shots_df.copy()

    # Resolvemos los nombres de columna sin importar mayúsculas/minúsculas —
    # 365Scores puede devolver "teamName", "team_name", "TeamName", etc.
    cols_map = {c.lower(): c for c in df.columns}
    resolved_outcome = cols_map.get(outcome_col.lower())
    resolved_team = cols_map.get(team_col.lower())

    if resolved_outcome is None:
        raise KeyError(
            f"No encontré la columna '{outcome_col}' en el shotmap. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    if resolved_team is None:
        raise KeyError(
            f"No encontré la columna '{team_col}' en el shotmap. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    outcome_col, team_col = resolved_outcome, resolved_team

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


def team_shot_summary(shots_df, team_col="teamName", xg_col="xg", xgot_col="xgot"):
    """Totales por equipo: tiros, tiros al arco, xG, xGOT — como el panel central del reporte."""
    df = shots_df.copy()
    cols = {c.lower(): c for c in df.columns}
    xg_col = cols.get(xg_col.lower())
    xgot_col = cols.get(xgot_col.lower())
    shottype_col = "shotType" if "shotType" in df.columns else None

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


def convert_goalmouth_to_plot(df, y_col="goalMouthY", z_col="goalMouthZ"):
    """
    Convierte las coordenadas crudas de arco (goalMouthY/Z, escala 0-100 aprox.
    tipo 365Scores) a coordenadas de ploteo sobre un arco dibujado en mplsoccer
    (ancho ~7.32m / alto ~2.44m, expresado en el sistema opta 0-100 horizontal).

    OJO: esta conversión está reconstruida a partir de los valores que se ven
    en tu log (ej. goalMouthY=52.0 -> goalMouthY_plot=35.78), no del código
    original. Si los valores no calzan visualmente al probarlo, es el primer
    lugar para ajustar la fórmula.
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
