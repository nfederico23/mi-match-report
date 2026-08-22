"""
Match Report Generator
-----------------------
App de Streamlit que arma un reporte visual de un partido (estilo whoscored/Opta)
usando LanusStats para scrapear datos de SofaScore o FotMob.

Cómo correrla localmente:
    pip install -r requirements.txt
    streamlit run app.py

NOTA IMPORTANTE:
El reporte de referencia (estilo "Scoresway") en realidad saca los datos de
365Scores, no de una fuente aparte. LanusStats soporta 365Scores directamente
(ls.ThreeSixFiveScores()), así que acá están las 3 fuentes: SofaScore, FotMob
y 365Scores.

Los nombres de columnas que devuelve cada función de LanusStats pueden variar
según la versión del paquete instalada. Si algo rompe, correr:
    st.write(df.columns.tolist())
en el bloque correspondiente para ver los nombres reales y ajustar.
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np

try:
    import LanusStats as ls
except ImportError:
    ls = None

try:
    from mplsoccer import Pitch
except ImportError:
    Pitch = None

from shot_utils import normalize_shotmap, team_shot_summary, convert_goalmouth_to_plot
from player_report import build_player_figure


st.set_page_config(page_title="Match Report", layout="wide")


# ----------------------------------------------------------------------------
# Sidebar - configuración
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

source = st.sidebar.radio("Fuente de datos", ["365Scores", "SofaScore", "FotMob"])

match_url = None
match_id = None

if source == "365Scores":
    match_url = st.sidebar.text_input(
        "URL del partido en 365Scores",
        placeholder="https://www.365scores.com/es-mx/football/match/copa-de-la-liga-profesional-7214/...#id=4033824",
    )
elif source == "SofaScore":
    match_url = st.sidebar.text_input(
        "URL del partido en SofaScore",
        placeholder="https://www.sofascore.com/arsenal-manchester-united/KR#id:11352532",
    )
else:
    match_id = st.sidebar.text_input(
        "Match ID de FotMob",
        placeholder="Ej: 4193851 (el número al final de la URL del partido)",
    )

home_name = st.sidebar.text_input("Nombre equipo local", "Local")
away_name = st.sidebar.text_input("Nombre equipo visitante", "Visitante")
home_color = st.sidebar.color_picker("Color local", "#22c55e")
away_color = st.sidebar.color_picker("Color visitante", "#ef4444")

st.sidebar.markdown("---")
report_type = st.sidebar.radio("Tipo de reporte", ["Equipo", "Jugador"])
player_name = None
if report_type == "Jugador":
    if source != "SofaScore":
        st.sidebar.warning("El reporte individual solo está armado para SofaScore por ahora "
                            "(es la única fuente con eventos de pases/conducciones/defensivas por jugador).")
    player_name = st.sidebar.text_input("Nombre del jugador (tal cual figura en SofaScore)")

generate = st.sidebar.button("🔎 Generar reporte", type="primary")

st.title("📊 Match Report Generator")
st.caption("Reporte de partido generado con LanusStats + Streamlit")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def safe_call(label, func, *args, **kwargs):
    """Ejecuta una función de scraping y muestra el error en pantalla si falla,
    sin tirar abajo el resto de la app."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.warning(f"No se pudo obtener: {label} → {e}")
        return None


def plot_shotmap(ax, shotmap_df, team_col_candidates, home_name, away_name):
    """Dibuja el mapa de tiros sobre una cancha horizontal partida en dos mitades."""
    if Pitch is None:
        ax.text(0.5, 0.5, "mplsoccer no instalado", ha="center")
        return

    pitch = Pitch(pitch_type="opta", pitch_color="#0e1117", line_color="white")
    pitch.draw(ax=ax)

    if shotmap_df is None or shotmap_df.empty:
        ax.set_title("Sin datos de tiros", color="white")
        return

    # Intentamos detectar las columnas de equipo / coordenadas / xg
    cols = {c.lower(): c for c in shotmap_df.columns}
    team_col = next((cols[c] for c in team_col_candidates if c in cols), None)
    x_col = next((cols[c] for c in ["x", "player_x", "shot_x"] if c in cols), None)
    y_col = next((cols[c] for c in ["y", "player_y", "shot_y"] if c in cols), None)
    xg_col = next((cols[c] for c in ["xg", "expectedgoals"] if c in cols), None)

    if not (x_col and y_col):
        ax.set_title("No se detectaron coordenadas de tiro", color="white")
        return

    if team_col:
        # El valor de team_col puede ser texto ("Sarmiento", "home") o numérico
        # (1/2, competitorNum) — no asumimos formato, solo tomamos los valores
        # únicos y les asignamos color local/visitante en orden.
        unique_teams = sorted(shotmap_df[team_col].dropna().unique(), key=str)
        color_by_team = {}
        if len(unique_teams) >= 1:
            color_by_team[unique_teams[0]] = home_color
        if len(unique_teams) >= 2:
            color_by_team[unique_teams[1]] = away_color

        for team_value, color in color_by_team.items():
            sub = shotmap_df[shotmap_df[team_col] == team_value]
            sizes = (sub[xg_col] * 400 + 40) if xg_col else 80
            ax.scatter(sub[x_col], sub[y_col], s=sizes, color=color, alpha=0.7, edgecolors="white", linewidths=0.5)
    else:
        sizes = (shotmap_df[xg_col] * 400 + 40) if xg_col else 80
        ax.scatter(shotmap_df[x_col], shotmap_df[y_col], s=sizes, color=home_color,
                   alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.set_title("Mapa de tiros", color="white")


def team_stats_bar(ax, stats_dict):
    """Panel tipo 'Match Stats': barras horizontales comparando dos equipos."""
    labels = list(stats_dict.keys())
    home_vals = [stats_dict[k][0] for k in labels]
    away_vals = [stats_dict[k][1] for k in labels]

    y = np.arange(len(labels))
    max_vals = [max(h, a, 1) for h, a in zip(home_vals, away_vals)]

    for i, (h, a, m) in enumerate(zip(home_vals, away_vals, max_vals)):
        ax.barh(i, -h / m, color=home_color, height=0.6)
        ax.barh(i, a / m, color=away_color, height=0.6)
        ax.text(-1.05, i, f"{h}", ha="right", va="center", color="white", fontsize=9)
        ax.text(1.05, i, f"{a}", ha="left", va="center", color="white", fontsize=9)
        ax.text(0, i, labels[i], ha="center", va="center", color="black", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"))

    ax.set_xlim(-1.3, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Match Stats", color="white")


# ----------------------------------------------------------------------------
# SofaScore report
# ----------------------------------------------------------------------------
def build_sofascore_report(match_url):
    if ls is None:
        st.error("LanusStats no está instalado en este entorno.")
        return

    sofascore = ls.SofaScore()

    shotmap = safe_call("mapa de tiros (SofaScore)", sofascore.get_match_shotmap, match_url)
    players_stats = safe_call("estadísticas de jugadores (SofaScore)", sofascore.get_players_match_stats, match_url)
    avg_positions = safe_call("posiciones promedio (SofaScore)", sofascore.get_players_average_positions, match_url)
    lineups = safe_call("lineups (SofaScore)", sofascore.get_lineups, match_url)

    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    fig.suptitle(f"{home_name}  vs  {away_name}", fontsize=22, color="white", fontweight="bold")

    # Shotmap
    ax_shot = fig.add_subplot(gs[0, 0:2])
    plot_shotmap(ax_shot, shotmap, ["team", "teamname", "side"], home_name, away_name)

    # Posiciones promedio (si vienen como lista [home_df, away_df])
    ax_pos = fig.add_subplot(gs[0, 2])
    ax_pos.set_facecolor("#0e1117")
    ax_pos.set_title("Posiciones promedio", color="white")
    if avg_positions and Pitch is not None:
        pitch = Pitch(pitch_type="opta", pitch_color="#0e1117", line_color="white")
        pitch.draw(ax=ax_pos)
        try:
            home_df, away_df = avg_positions[0], avg_positions[1]
            for df, color in [(home_df, home_color), (away_df, away_color)]:
                cols = {c.lower(): c for c in df.columns}
                x_col = next((cols[c] for c in ["averagex", "x"] if c in cols), None)
                y_col = next((cols[c] for c in ["averagey", "y"] if c in cols), None)
                if x_col and y_col:
                    ax_pos.scatter(df[x_col], df[y_col], s=300, color=color, edgecolors="white", zorder=3)
        except Exception as e:
            st.warning(f"No se pudo graficar posiciones promedio: {e}")
    else:
        ax_pos.text(0.5, 0.5, "Sin datos", ha="center", color="white", transform=ax_pos.transAxes)
    ax_pos.axis("off")

    # Match stats agregando players_stats (si están disponibles)
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.set_facecolor("#0e1117")
    if players_stats:
        try:
            home_df, away_df = players_stats[0], players_stats[1]
            numeric_cols = [c for c in home_df.columns if pd.api.types.is_numeric_dtype(home_df[c])]
            candidate_stats = [c for c in numeric_cols if any(
                k in c.lower() for k in ["pass", "tackle", "shot", "duel", "interception"]
            )][:6]
            stats_dict = {
                c: (round(home_df[c].sum(), 1), round(away_df[c].sum(), 1)) for c in candidate_stats
            }
            if stats_dict:
                team_stats_bar(ax_stats, stats_dict)
            else:
                ax_stats.text(0.5, 0.5, "No se detectaron columnas de stats", ha="center",
                               color="white", transform=ax_stats.transAxes)
                ax_stats.axis("off")
        except Exception as e:
            st.warning(f"No se pudo armar match stats: {e}")
            ax_stats.axis("off")
    else:
        ax_stats.axis("off")

    # Lineups como texto simple
    ax_lineup = fig.add_subplot(gs[1, 1:])
    ax_lineup.set_facecolor("#0e1117")
    ax_lineup.axis("off")
    ax_lineup.set_title("Formaciones / Lineups", color="white", loc="left")
    if lineups is not None:
        ax_lineup.text(0.02, 0.9, str(lineups)[:800], color="white", fontsize=7,
                        va="top", wrap=True, transform=ax_lineup.transAxes)

    st.pyplot(fig)

    with st.expander("Ver dataframes crudos (para debug / futuras mejoras)"):
        st.write("Shotmap", shotmap)
        st.write("Players stats", players_stats)
        st.write("Average positions", avg_positions)
        st.write("Lineups", lineups)


# ----------------------------------------------------------------------------
# FotMob report
# ----------------------------------------------------------------------------
def build_fotmob_report(match_id):
    if ls is None:
        st.error("LanusStats no está instalado en este entorno.")
        return

    fotmob = ls.FotMob()

    shotmap = safe_call("mapa de tiros (FotMob)", fotmob.get_match_shotmap, int(match_id))
    general_stats = safe_call("estadísticas generales (FotMob)", fotmob.get_general_match_stats, int(match_id))

    fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    fig.suptitle(f"{home_name}  vs  {away_name}", fontsize=22, color="white", fontweight="bold")

    ax_shot = fig.add_subplot(gs[0, 0])
    plot_shotmap(ax_shot, shotmap, ["team", "teamname", "side", "h_a"], home_name, away_name)

    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.set_facecolor("#0e1117")
    if general_stats is not None:
        try:
            cols = {c.lower(): c for c in general_stats.columns}
            stat_col = next((cols[c] for c in ["stat", "title", "statname"] if c in cols), general_stats.columns[0])
            home_col = next((cols[c] for c in ["home", "homevalue"] if c in cols), general_stats.columns[1])
            away_col = next((cols[c] for c in ["away", "awayvalue"] if c in cols), general_stats.columns[2])
            stats_dict = {
                str(row[stat_col]): (row[home_col], row[away_col])
                for _, row in general_stats.iterrows()
                if str(row[home_col]).replace(".", "", 1).isdigit()
            }
            team_stats_bar(ax_stats, stats_dict if stats_dict else {"Sin datos numéricos": (0, 0)})
        except Exception as e:
            st.warning(f"No se pudo armar match stats: {e}")
            ax_stats.axis("off")
    else:
        ax_stats.axis("off")

    st.pyplot(fig)

    st.info("💡 Para el Match Momentum (xT), FotMob trae una función lista en LanusStats:")
    st.code("ls.visualizations.fotmob_match_momentum_plot(match_id=" + str(match_id) + ")", language="python")
    if st.checkbox("Generar Match Momentum también"):
        try:
            ls.visualizations.fotmob_match_momentum_plot(match_id=int(match_id), save_fig=False)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.warning(f"No se pudo generar el match momentum: {e}")

    with st.expander("Ver dataframes crudos (para debug / futuras mejoras)"):
        st.write("Shotmap", shotmap)
        st.write("General stats", general_stats)


# ----------------------------------------------------------------------------
# 365Scores report
# ----------------------------------------------------------------------------
def build_365_report(match_url):
    if ls is None:
        st.error("LanusStats no está instalado en este entorno.")
        return

    ts = ls.ThreeSixFiveScores()

    shotmap = safe_call("mapa de tiros (365Scores)", ts.get_match_shotmap, match_url)
    general_stats = safe_call("estadísticas generales (365Scores)", ts.get_match_general_stats, match_url)
    time_stats = safe_call("estadísticas de tiempo (365Scores)", ts.get_match_time_stats, match_url)

    normalized, grouped, summary = None, None, None
    if shotmap is not None:
        try:
            normalized, grouped = normalize_shotmap(shotmap)
            summary = team_shot_summary(normalized)
        except Exception as e:
            st.warning(f"No se pudo normalizar el shotmap: {e}")

    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle(f"{home_name}  vs  {away_name}", fontsize=22, color="white", fontweight="bold")

    # Shotmap normalizado
    ax_shot = fig.add_subplot(gs[0, 0:2])
    plot_shotmap(ax_shot, normalized if normalized is not None else shotmap,
                 ["competitornum", "teamname", "team", "side"], home_name, away_name)

    # Panel central: resumen tiros/xG/xGOT por equipo (como el panel del PDF)
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.set_facecolor("#0e1117")
    ax_summary.axis("off")
    ax_summary.set_title("Remates", color="white")
    if summary:
        # La clave de "summary" puede ser 1/2, "home"/"away", etc. según qué
        # columna se haya usado como equipo — la mapeamos a los nombres que
        # puso el usuario, asumiendo que el primer valor (orden ascendente/
        # alfabético) es el local. Si no coincide, es fácil de ajustar acá.
        raw_keys = sorted(summary.keys(), key=str)
        name_map = {}
        if len(raw_keys) == 2:
            name_map = {raw_keys[0]: home_name, raw_keys[1]: away_name}

        y = 0.9
        for i, (team, s) in enumerate(summary.items()):
            label = name_map.get(team, str(team))
            color = home_color if i == 0 else away_color
            ax_summary.text(0.5, y, label, color=color, fontsize=12, fontweight="bold",
                             ha="center", transform=ax_summary.transAxes)
            y -= 0.08
            for k, v in s.items():
                ax_summary.text(0.5, y, f"{k}: {v}", color="white", fontsize=10,
                                 ha="center", transform=ax_summary.transAxes)
                y -= 0.07
            y -= 0.05
    else:
        ax_summary.text(0.5, 0.5, "Sin datos", ha="center", color="white", transform=ax_summary.transAxes)

    # Match stats (official-like) si vienen en general_stats
    ax_stats = fig.add_subplot(gs[1, 0:2])
    ax_stats.set_facecolor("#0e1117")
    if general_stats is not None:
        try:
            cols = {c.lower(): c for c in general_stats.columns}
            stat_col = next((cols[c] for c in ["stat", "title", "statname"] if c in cols), general_stats.columns[0])
            home_col = next((cols[c] for c in ["home", "homevalue"] if c in cols), general_stats.columns[1])
            away_col = next((cols[c] for c in ["away", "awayvalue"] if c in cols), general_stats.columns[2])
            stats_dict = {
                str(row[stat_col]): (row[home_col], row[away_col])
                for _, row in general_stats.iterrows()
                if str(row[home_col]).replace(".", "", 1).isdigit()
            }
            team_stats_bar(ax_stats, stats_dict if stats_dict else {"Sin datos": (0, 0)})
        except Exception as e:
            st.warning(f"No se pudo armar match stats: {e}")
            ax_stats.axis("off")
    else:
        ax_stats.axis("off")

    # Arco: goles y atajadas (goalmouth plot) — reconstruido del log
    ax_goal = fig.add_subplot(gs[1, 2])
    ax_goal.set_facecolor("#0e1117")
    ax_goal.set_title("Goles y atajadas", color="white")
    if normalized is not None and "goalMouthY" in normalized.columns:
        try:
            goalmouth_df = normalized[normalized["shotType"].isin(["goal", "save"])]
            goalmouth_df = convert_goalmouth_to_plot(goalmouth_df)
            for _, row in goalmouth_df.iterrows():
                color = home_color if row["shotType"] == "goal" else "white"
                ax_goal.scatter(row["goalMouthY_plot"], row["goalMouthZ_plot"], s=120, color=color,
                                 edgecolors="black", zorder=3)
            ax_goal.set_xlim(0, 100)
            ax_goal.set_ylim(0, 40)
        except Exception as e:
            st.warning(f"No se pudo graficar el arco: {e}")
    else:
        ax_goal.text(0.5, 0.5, "Sin datos de arco", ha="center", color="white", transform=ax_goal.transAxes)
    ax_goal.set_xticks([])
    ax_goal.set_yticks([])

    st.pyplot(fig)

    with st.expander("Ver dataframes crudos (para debug / futuras mejoras)"):
        st.write("Shotmap normalizado", normalized)
        st.write("Resumen agrupado", grouped)
        st.write("Resumen por equipo", summary)
        st.write("General stats", general_stats)
        st.write("Time stats", time_stats)


# ----------------------------------------------------------------------------
# Reporte individual de jugador (solo SofaScore por ahora)
# ----------------------------------------------------------------------------
def build_individual_report(match_url, player):
    if ls is None:
        st.error("LanusStats no está instalado en este entorno.")
        return

    sofascore = ls.SofaScore()

    events = safe_call(
        f"eventos de {player}",
        sofascore.get_player_match_events,
        match_url, player, events=["passes", "ball-carries", "dribbles", "defensive"],
    )
    heatmap_img = safe_call(f"mapa de calor de {player}", sofascore.get_player_heatmap, match_url, player)

    if events is None:
        st.error("No se pudieron obtener eventos para este jugador. Revisá que el nombre "
                 "esté escrito exactamente como en SofaScore.")
        return

    # get_player_match_events puede devolver un dict {tipo: df} o una lista de dfs
    # según versión — normalizamos a dict.
    if isinstance(events, list):
        keys = ["passes", "ball-carries", "dribbles", "defensive"]
        events = {k: v for k, v in zip(keys, events)}

    fig = build_player_figure(
        player_name=player,
        opponent_name=away_name if away_name != "Visitante" else "Rival",
        minutes="-",
        events=events,
        heatmap_img=heatmap_img,
        team_color=home_color,
    )
    st.pyplot(fig)

    with st.expander("Ver dataframes crudos (para debug / futuras mejoras)"):
        for k, v in events.items():
            st.write(k, v)
        st.write("Heatmap", heatmap_img)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if generate:
    if ls is None:
        st.error("Falta instalar LanusStats: pip install LanusStats")
    elif report_type == "Jugador":
        if source != "SofaScore":
            st.error("El reporte individual solo está armado para SofaScore por ahora.")
        elif not match_url or not player_name:
            st.warning("Pegá la URL del partido de SofaScore y el nombre del jugador en la barra lateral.")
        else:
            with st.spinner(f"Scrapeando eventos de {player_name}..."):
                build_individual_report(match_url, player_name)
    elif source == "365Scores":
        if not match_url:
            st.warning("Pegá la URL del partido de 365Scores en la barra lateral.")
        else:
            with st.spinner("Scrapeando 365Scores..."):
                build_365_report(match_url)
    elif source == "SofaScore":
        if not match_url:
            st.warning("Pegá la URL del partido de SofaScore en la barra lateral.")
        else:
            with st.spinner("Scrapeando SofaScore..."):
                build_sofascore_report(match_url)
    else:
        if not match_id:
            st.warning("Pegá el Match ID de FotMob en la barra lateral.")
        else:
            with st.spinner("Scrapeando FotMob..."):
                build_fotmob_report(match_id)
else:
    st.info("⬅️ Configurá la fuente y el partido en la barra lateral, y tocá **Generar reporte**.")
