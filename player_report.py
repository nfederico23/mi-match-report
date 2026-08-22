"""
player_report.py
-----------------
Reporte individual de jugador (estilo el PDF "O. Cortes / F. Kalinger / ...")
usando SofaScore vía LanusStats:
  - get_player_match_events(match_url, player, events=[...]) -> pases, conducciones,
    dribbles, acciones defensivas
  - get_player_heatmap(match_url, player) -> imagen PIL

Es la fuente con más detalle a nivel jugador dentro de LanusStats; por eso el
reporte individual está armado solo para SofaScore por ahora.

El layout usa pitch.grid() de mplsoccer en vez de GridSpec a mano: acomoda
varias canchas en una figura con espaciado automático y ejes de título/pie
prolijos (ver https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_grid.html).
Los paneles que no son cancha (datos, mapa de calor) se arman limpiando esa
misma cancha con ax.cla() y dibujando encima.

Igual que en shot_utils.py: esto no está probado contra la API real (sandbox
sin salida a internet), así que los nombres de columna son la mejor
aproximación a partir de la documentación — ajustar en base a lo que devuelva
`get_player_match_events` real la primera vez que se corra.
"""

try:
    from mplsoccer import Pitch
except ImportError:
    Pitch = None


PITCH_COLOR = "#0e1117"
LINE_COLOR = "#666666"


def _plot_lines(ax, df, x1, y1, x2, y2, color, success_col=None):
    """Dibuja líneas evento->destino (pases, conducciones) sobre la cancha."""
    if df is None or df.empty:
        return
    cols = {c.lower(): c for c in df.columns}
    x1c, y1c = cols.get(x1), cols.get(y1)
    x2c, y2c = cols.get(x2), cols.get(y2)
    if not all([x1c, y1c, x2c, y2c]):
        return
    for _, row in df.iterrows():
        c = color
        if success_col and success_col in cols:
            c = color if row.get(cols[success_col]) else "#888888"
        ax.plot([row[x1c], row[x2c]], [row[y1c], row[y2c]], color=c, alpha=0.7, linewidth=1.2)
        ax.scatter(row[x1c], row[y1c], color=c, s=15, zorder=3)


def _plot_points(ax, df, x_col, y_col, color, marker="o"):
    if df is None or df.empty:
        return
    cols = {c.lower(): c for c in df.columns}
    xc, yc = cols.get(x_col), cols.get(y_col)
    if not xc or not yc:
        return
    ax.scatter(df[xc], df[yc], color=color, s=60, marker=marker, edgecolors="white", linewidths=0.5, zorder=3)


def _clear_to_blank(ax, title):
    """Convierte un ax de cancha (ya dibujado por pitch.grid) en un panel de texto/imagen liso."""
    ax.cla()
    ax.set_facecolor(PITCH_COLOR)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=11, loc="left")


def build_player_figure(player_name, opponent_name, minutes, events, heatmap_img,
                         team_color="#22c55e"):
    """
    events: dict con hasta 4 dataframes -> {'passes': df, 'ball-carries': df,
            'dribbles': df, 'defensive': df} (lo que devuelva get_player_match_events)
    heatmap_img: imagen PIL de get_player_heatmap, o None

    Layout (grilla 2x3, vía pitch.grid):
      [Mapa de pases]   [Conducciones]     [Datos]
      [Pases recibidos] [Acc. defensivas]  [Mapa de calor]
    """
    if Pitch is None:
        raise ImportError("mplsoccer no está instalado (pip install mplsoccer)")

    pitch = Pitch(pitch_type="opta", pitch_color=PITCH_COLOR, line_color=LINE_COLOR, linewidth=1)
    fig, axs = pitch.grid(
        nrows=2, ncols=3,
        figheight=9,
        title_height=0.12, title_space=0.01,
        endnote_height=0.04, endnote_space=0.01,
        grid_height=0.80, grid_width=0.95,
        space=0.08,
        axis=False,
    )
    fig.set_facecolor("#f5f5f5")

    # Título (ax dedicado, ya alineado por pitch.grid)
    axs["title"].text(0.01, 0.65, player_name, fontsize=22, fontweight="bold",
                       ha="left", va="center", transform=axs["title"].transAxes)
    axs["title"].text(0.01, 0.15,
                       f"vs {opponent_name}  |  Minutos {minutes}  |  Reporte generado con LanusStats",
                       fontsize=10, color="#444444", ha="left", va="center",
                       transform=axs["title"].transAxes)

    axs["endnote"].text(0.99, 0.5, "Adaptación propia — modelo original de @adnaaan433",
                         fontsize=8, color="#666666", ha="right", va="center",
                         transform=axs["endnote"].transAxes)

    passes = events.get("passes")
    carries = events.get("ball-carries")
    defensive = events.get("defensive")

    grid = axs["pitch"]  # array 2x3 de axes de cancha ya dibujadas
    ax_passes, ax_carries, ax_data = grid[0, 0], grid[0, 1], grid[0, 2]
    ax_received, ax_defensive, ax_heat = grid[1, 0], grid[1, 1], grid[1, 2]

    # Mapa de pases
    ax_passes.set_title("Mapa de pases", color="white", fontsize=11, loc="left")
    _plot_lines(ax_passes, passes, "x", "y", "endx", "endy", team_color, success_col="accurate")

    # Conducciones
    ax_carries.set_title("Conducciones", color="white", fontsize=11, loc="left")
    _plot_lines(ax_carries, carries, "x", "y", "endx", "endy", "#f59e0b")

    # Acciones defensivas
    ax_defensive.set_title("Acciones defensivas", color="white", fontsize=11, loc="left")
    _plot_points(ax_defensive, defensive, "x", "y", "#22c55e", marker="X")

    # Pases recibidos (placeholder — requiere eventos de todo el equipo)
    ax_received.set_title("Pases recibidos", color="white", fontsize=11, loc="left")
    ax_received.text(0.5, 0.02, "Requiere eventos del equipo completo\n(no incluido en get_player_match_events)",
                      ha="center", va="bottom", fontsize=7, color="white", transform=ax_received.transAxes)

    # Datos (texto) — limpiamos la cancha dibujada y dejamos un panel liso
    _clear_to_blank(ax_data, "Datos")
    lines = []
    if passes is not None and not passes.empty:
        cols = {c.lower(): c for c in passes.columns}
        total = len(passes)
        acc = passes[cols["accurate"]].sum() if "accurate" in cols else None
        pct = f"{acc/total*100:.1f}%" if acc is not None and total else "-"
        lines.append(f"Pases efectivos: {acc if acc is not None else '-'}/{total} ({pct})")
    if carries is not None and not carries.empty:
        lines.append(f"Conducciones: {len(carries)}")
    if defensive is not None and not defensive.empty:
        lines.append(f"Acciones defensivas: {len(defensive)}")
    if not lines:
        lines = ["Sin datos de eventos para este jugador."]
    ax_data.text(0.02, 0.85, "\n".join(lines), fontsize=10, va="top", color="white",
                 transform=ax_data.transAxes)

    # Mapa de calor — limpiamos la cancha dibujada y mostramos la imagen PIL
    _clear_to_blank(ax_heat, "Toques y mapa de calor")
    if heatmap_img is not None:
        ax_heat.imshow(heatmap_img)
    else:
        ax_heat.text(0.5, 0.5, "Sin mapa de calor", ha="center", va="center",
                      color="white", transform=ax_heat.transAxes)

    return fig
