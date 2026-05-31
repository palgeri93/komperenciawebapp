import io
import re
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Kompetenciamérés PDF generátor", layout="wide")

ANGOL_SZINT_MAP = {"pre-a1": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6}
ANGOL_SZINT_LABELS = {
    0: "0. szint - pre-A1 alatti szint",
    1: "1. szint - pre-A1 szint",
    2: "2. szint - A1 szint",
    3: "3. szint - A2 szint",
    4: "4. szint - B1 szint",
    5: "5. szint - B2 szint",
    6: "6. szint - C1 szint vagy fölötte",
}
BASE_COLUMNS = ["Évfolyam", "Tanulócsoportok", "Mérési azonosító", "Név"]
DEFAULT_AREAS = [
    "Szövegértés", "Matematika", "Természettudomány", "Angol nyelv",
    "Német nyelv", "Digitális kultúra", "Történelem"
]
CHANGE_ORDER = [
    "Nincs mindkét eredmény",
    "Jelentős -",
    "Mérsékelt -",
    "Elhanyagolható",
    "Mérsékelt +",
    "Jelentős +",
]
CHANGE_COLORS = {
    "Nincs mindkét eredmény": "#bdbdbd",
    "Jelentős -": "#8B0000",
    "Mérsékelt -": "#FF6347",
    "Elhanyagolható": "#9e9e9e",
    "Mérsékelt +": "#90EE90",
    "Jelentős +": "#006400",
}
LEVEL_COLORS = ["#d9d9d9", "#dff7d8", "#b9efad", "#62d66f", "#3fc76b", "#22bf73", "#17aa67", "#0c8f58"]


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Név" not in df.columns and "NEVEK" in df.columns:
        df = df.rename(columns={"NEVEK": "Név"})
    return df


def decode_angol_szint(value: object) -> int:
    text = str(value).lower().replace("–", "-")
    if "pre-a1" in text and "alatti" in text:
        return 0
    for key, level in ANGOL_SZINT_MAP.items():
        if key in text:
            return level
    return 0


def decode_numeric_level(value: object) -> float:
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else 0.0


def categorize_change(value: float) -> tuple[str, str]:
    if value <= -100:
        return "#8B0000", "Jelentős romlás"
    if -99 <= value <= -40:
        return "#FF6347", "Mérsékelt romlás"
    if -39 <= value < 0:
        return "gray", "Elhanyagolható romlás"
    if 0 <= value <= 39:
        return "gray", "Elhanyagolható javulás"
    if 40 <= value <= 100:
        return "#90EE90", "Mérsékelt javulás"
    if value > 100:
        return "#006400", "Jelentős javulás"
    return "black", "Ismeretlen"


def short_change_category(row: pd.Series) -> str:
    p1 = pd.to_numeric(row.get("Képességpont_1"), errors="coerce")
    p2 = pd.to_numeric(row.get("Képességpont_2"), errors="coerce")
    if pd.isna(p1) and pd.isna(p2):
        return "Nincs mindkét eredmény"
    value = float(row.get("Képességpont változás", 0) or 0)
    if value <= -100:
        return "Jelentős -"
    if -99 <= value <= -40:
        return "Mérsékelt -"
    if -39 <= value <= 39:
        return "Elhanyagolható"
    if 40 <= value <= 100:
        return "Mérsékelt +"
    if value > 100:
        return "Jelentős +"
    return "Elhanyagolható"


def get_sheet_names(uploaded_file) -> list[str]:
    uploaded_file.seek(0)
    names = pd.ExcelFile(uploaded_file).sheet_names
    uploaded_file.seek(0)
    return names


def detect_areas(uploaded_file, sheet_name: str, area_row_index: int = 0) -> dict[str, int]:
    uploaded_file.seek(0)
    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=3)
    uploaded_file.seek(0)
    areas: dict[str, int] = {}
    for col_index, value in enumerate(preview.iloc[area_row_index].tolist()):
        name = clean_text(value)
        if name and name.lower() not in {"nan", "none"}:
            areas[name] = col_index
    return areas


def read_selected_area(uploaded_file, sheet_name: str, area_name: str, header_row_index: int = 2) -> pd.DataFrame:
    """
    Egy kiválasztott mérési terület beolvasása oszloppozíciók alapján.

    Fontos javítás: az Excelben több mérési terület alatt is ugyanazok az oszlopnevek
    szerepelnek (Képességpont, Képességszint stb.). Pandasban a név szerinti
    kiválasztás ilyenkor az összes azonos nevű oszlopot visszaadhatja, ezért itt
    kizárólag oszloppozícióval dolgozunk. Ez javítja a 4. ábra hibáját is, ahol
    korábban 100+ hamis „szint” jelent meg a jelmagyarázatban.
    """
    uploaded_file.seek(0)
    full_headerless = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
    uploaded_file.seek(0)
    area_map = detect_areas(uploaded_file, sheet_name, 0)
    uploaded_file.seek(0)
    if area_name not in area_map:
        raise ValueError(f"Nem találom ezt a mérési területet: {area_name}")

    start = int(area_map[area_name])
    next_starts = sorted(idx for name, idx in area_map.items() if idx > start)
    end = int(next_starts[0]) if next_starts else full_headerless.shape[1]

    headers = [clean_text(x) for x in full_headerless.iloc[header_row_index].tolist()]
    data = full_headerless.iloc[header_row_index + 1 :].reset_index(drop=True)

    # Alapadatok: az első négy oszlop fixen Évfolyam, Tanulócsoportok, Mérési azonosító, Név/NEVEK.
    result = pd.DataFrame()
    for pos, out_name in [(0, "Évfolyam"), (1, "Tanulócsoportok"), (2, "Mérési azonosító"), (3, "Név")]:
        if pos < data.shape[1]:
            result[out_name] = data.iloc[:, pos]

    # A kiválasztott mérési terület blokkjának oszlopai.
    area = data.iloc[:, start:end].copy()
    area_headers = headers[start:end]
    if area.shape[1] < 4:
        raise ValueError("A kiválasztott területnél nincs elég adat a diagramhoz.")

    # A jelentésekben az első négy területi oszlop: előző tanévi pont/szint, előzetes pont/szint.
    result["Képességpont_1"] = area.iloc[:, 0]
    result["Képességszint"] = area.iloc[:, 1]
    result["Képességpont_2"] = area.iloc[:, 2]
    result["Képességszint.1"] = area.iloc[:, 3]

    change_col = None
    for idx, col_name in enumerate(area_headers):
        col_text = str(col_name).lower()
        if "változás" in col_text and "%" not in col_text:
            change_col = area.iloc[:, idx]
            break
    if change_col is None:
        result["Képességpont változás"] = pd.to_numeric(result["Képességpont_2"], errors="coerce") - pd.to_numeric(result["Képességpont_1"], errors="coerce")
    else:
        result["Képességpont változás"] = change_col
    return result


def get_classes(df: pd.DataFrame) -> list[str]:
    if "Tanulócsoportok" not in df.columns:
        return ["Összes tanuló"]
    classes = sorted([x for x in df["Tanulócsoportok"].dropna().map(clean_text).unique() if x])
    return ["Összes tanuló"] + classes


def prepare_dataframe(df: pd.DataFrame, selected_class: str, angol: bool) -> pd.DataFrame:
    df = df.copy()
    if selected_class != "Összes tanuló" and "Tanulócsoportok" in df.columns:
        df = df[df["Tanulócsoportok"].map(clean_text).eq(selected_class)]
    if angol:
        df["Szint_1"] = df["Képességszint"].apply(decode_angol_szint)
        df["Szint_2"] = df["Képességszint.1"].apply(decode_angol_szint)
    else:
        df["Szint_1"] = df["Képességszint"].apply(decode_numeric_level)
        df["Szint_2"] = df["Képességszint.1"].apply(decode_numeric_level)
    df["Képességpont változás"] = pd.to_numeric(df["Képességpont változás"], errors="coerce").fillna(0)
    df["Név"] = df["Név"].astype(str).apply(lambda x: " ".join(x.split()))
    df = df[df["Név"].str.lower().ne("nan") & df["Név"].ne("")]
    df["Változás kategória"] = df.apply(short_change_category, axis=1)
    return df.sort_values(by="Képességpont változás")


def change_summary(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    counts = df["Változás kategória"].value_counts().reindex(CHANGE_ORDER, fill_value=0)
    total = max(len(df), 1)
    percent = (counts / total * 100).round(1)
    return pd.DataFrame([
        [group_label] + counts.astype(int).tolist(),
        ["Tanulók aránya"] + [f"{p:.1f}%".replace(".", ",") for p in percent.tolist()],
        ["Összesen"] + counts.astype(int).tolist(),
        ["Tanulók aránya"] + [f"{p:.1f}%".replace(".", ",") for p in percent.tolist()],
    ], columns=["Tanulócsoportok"] + CHANGE_ORDER)




def change_name_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Név szerinti bontás a változás mértéke szerint, a Word-jelentéshez is másolható formában."""
    mapping = {
        "Nincs mindkét eredmény": "Nincs mindkét eredmény",
        "Jelentős -": "Jelentős romlás",
        "Mérsékelt -": "Mérsékelt romlás",
        "Elhanyagolható": "Elhanyagolható változás",
        "Mérsékelt +": "Mérsékelt javulás",
        "Jelentős +": "Jelentős javulás",
    }
    groups = {label: [] for label in mapping.values()}
    for _, row in df.sort_values("Képességpont változás").iterrows():
        short = row.get("Változás kategória", "Elhanyagolható")
        label = mapping.get(short, "Elhanyagolható változás")
        name = clean_text(row.get("Név", ""))
        if name:
            groups[label].append(name)
    return groups


def change_names_text(df: pd.DataFrame) -> str:
    groups = change_name_groups(df)
    lines = []
    templates = [
        ("Jelentős romlás", "Jelentős romlás {count} tanulónál történt: {names}"),
        ("Mérsékelt romlás", "Mérsékelt romlás {count} főnél történt: {names}"),
        ("Elhanyagolható változás", "Elhanyagolható a változás {count} tanuló esetében: {names}"),
        ("Mérsékelt javulás", "Mérsékelt javulást mutat {count} tanuló: {names}"),
        ("Jelentős javulás", "Jelentős mértékű javulást mutat {count} fő: {names}"),
        ("Nincs mindkét eredmény", "Nincs mindkét eredménye {count} tanulónak: {names}"),
    ]
    for label, template in templates:
        names = groups.get(label, [])
        if not names and label == "Nincs mindkét eredmény":
            continue
        lines.append(template.format(count=len(names), names=", ".join(names) if names else "-"))
    return "\n".join(lines)


def level_distribution(df: pd.DataFrame, column: str, max_level: int) -> dict[int, int]:
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    return {level: int((vals == level).sum()) for level in range(0, max_level + 1)}


def add_change_chart(pdf: PdfPages, df: pd.DataFrame, osztaly: str, terulet: str) -> None:
    if df.empty or df["Képességpont változás"].abs().sum() <= 0:
        return
    colors, _ = zip(*[categorize_change(v) for v in df["Képességpont változás"]])
    plt.figure(figsize=(14, 6))
    bars = plt.bar(df["Név"], df["Képességpont változás"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom" if height >= 0 else "top", fontsize=8)
    plt.title(f"{osztaly} – {terulet} – Képességpont változás")
    plt.xlabel("Tanuló neve")
    plt.ylabel("Képességpont változás")
    plt.xticks(rotation=45, ha="right")
    legend_labels = {"Jelentős romlás": "#8B0000", "Mérsékelt romlás": "#FF6347", "Elhanyagolható romlás/javulás": "gray", "Mérsékelt javulás": "#90EE90", "Jelentős javulás": "#006400"}
    patches = [plt.Line2D([0], [0], color=color, lw=8, label=label) for label, color in legend_labels.items()]
    plt.legend(handles=patches, title="Kategóriák")
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def add_change_summary_table(pdf: PdfPages, df: pd.DataFrame, osztaly: str, terulet: str) -> None:
    table_df = change_summary(df, osztaly)
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.axis("off")
    ax.set_title("Változás mértéke az elemzésbe bevont csoportok esetében", loc="left", fontsize=14, fontweight="bold", pad=18)
    ax.text(0.5, 0.90, terulet, ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc="center", bbox=[0.02, 0.48, 0.96, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f3f3f3")
        if row in [1, 3]:
            cell.set_facecolor("#eeeeee")

    # Név szerinti felsorolás ugyanarra az oldalra kerül, a Word-jelentés 3. ábra alatti részéhez hasonlóan.
    y = 0.39
    for raw_line in change_names_text(df).splitlines():
        wrapped = textwrap.wrap(raw_line, width=125) or [raw_line]
        for line in wrapped:
            ax.text(0.02, y, line, ha="left", va="top", fontsize=10.5, transform=ax.transAxes)
            y -= 0.055
        y -= 0.012
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def add_level_distribution_chart(pdf: PdfPages, df: pd.DataFrame, osztaly: str, terulet: str, angol: bool) -> None:
    if df.empty:
        return
    max_level = int(6 if angol else 7)
    rows = [("2024/2025-ös tanév", "Szint_1"), ("2025/2026-os tanév előzetes eredmény", "Szint_2")]
    fig, ax = plt.subplots(figsize=(14, 6.0))
    y_positions = np.arange(len(rows))
    total = max(len(df), 1)
    present_levels = set()

    for y, (label, col) in zip(y_positions, rows):
        left = 0.0
        dist = level_distribution(df, col, max_level)
        for level in range(0, max_level + 1):
            count = dist[level]
            if count == 0:
                continue
            present_levels.add(level)
            pct = count / total * 100
            color = LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)]
            ax.barh(y, pct, left=left, height=0.46, color=color, edgecolor="white")
            label_text = f"{pct:.1f} %".replace(".", ",")

            # Minden nem nulla szegmens kap százalékfeliratot.
            # A nagyon keskeny szegmenseknél a feliratot a sávon kívülre tesszük,
            # különben a szöveg vagy lemaradna, vagy olvashatatlanul összecsúszna.
            center = left + pct / 2
            if pct >= 7:
                ax.text(center, y, label_text, ha="center", va="center", fontsize=10, fontweight="bold")
            elif pct >= 3:
                ax.text(center, y, label_text, ha="center", va="center", fontsize=8.5, fontweight="bold")
            else:
                offset_y = -0.34 if y == 0 else 0.34
                label_x = min(max(center, 2.0), 98.0)
                ax.annotate(
                    label_text,
                    xy=(center, y),
                    xytext=(label_x, y + offset_y),
                    textcoords="data",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "black", "shrinkA": 0, "shrinkB": 0},
                    clip_on=False,
                )
            left += pct

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tanulók aránya")
    ax.set_title(f"{osztaly} – {terulet} – Mérési szintek megoszlása", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{x},0 %" for x in [0, 20, 40, 60, 80, 100]])

    legend_levels = [level for level in range(0, max_level + 1) if level in present_levels]
    handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[min(level, len(LEVEL_COLORS)-1)], lw=8,
                   label=("Nincs eredmény" if level == 0 else f"{level}. szint"))
        for level in legend_levels
    ]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=min(len(handles), 8))
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.23, bottom=0.25, right=0.98, top=0.86)
    pdf.savefig(fig)
    plt.close(fig)


def add_level_chart(pdf: PdfPages, df: pd.DataFrame, alapszint: int, osztaly: str, terulet: str, angol: bool) -> None:
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, df["Szint_1"], width, label="2024/2025")
    ax.bar(x + width / 2, df["Szint_2"], width, label="2025/2026 előzetes")

    max_data_level = int(max(df["Szint_1"].max(), df["Szint_2"].max(), alapszint, 6 if angol else 7))
    ax.set_ylim(0, max_data_level + 0.8)
    if alapszint > 0:
        ax.axhline(alapszint, color="red", linestyle="--", linewidth=2, label=f"Alapszint: {alapszint}", zorder=3)

    ax.set_title(f"{osztaly} – {terulet} – Képességszint változás")
    ax.set_xlabel("Tanuló neve")
    ax.set_ylabel("Képességszint")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Név"], rotation=45, ha="right")
    if angol:
        ticks = sorted(ANGOL_SZINT_LABELS.keys())
        ax.set_yticks(ticks)
        ax.set_yticklabels([ANGOL_SZINT_LABELS[y] for y in ticks])
    else:
        ax.set_yticks(np.arange(0, max_data_level + 1, 1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.08, bottom=0.28, right=0.98, top=0.90)
    pdf.savefig(fig)
    plt.close(fig)



def make_change_chart_fig(df: pd.DataFrame, osztaly: str, terulet: str):
    if df.empty or df["Képességpont változás"].abs().sum() <= 0:
        return None
    colors, _ = zip(*[categorize_change(v) for v in df["Képességpont változás"]])
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(df["Név"], df["Képességpont változás"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom" if height >= 0 else "top", fontsize=8)
    ax.set_title(f"{osztaly} – {terulet} – Képességpont változás")
    ax.set_xlabel("Tanuló neve")
    ax.set_ylabel("Képességpont változás")
    ax.set_xticklabels(df["Név"], rotation=45, ha="right")
    legend_labels = {"Jelentős romlás": "#8B0000", "Mérsékelt romlás": "#FF6347", "Elhanyagolható romlás/javulás": "gray", "Mérsékelt javulás": "#90EE90", "Jelentős javulás": "#006400"}
    patches = [plt.Line2D([0], [0], color=color, lw=8, label=label) for label, color in legend_labels.items()]
    ax.legend(handles=patches, title="Kategóriák")
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    return fig


def make_change_summary_fig(df: pd.DataFrame, osztaly: str, terulet: str):
    table_df = change_summary(df, osztaly)
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.axis("off")
    ax.set_title("Változás mértéke az elemzésbe bevont csoportok esetében", loc="left", fontsize=14, fontweight="bold", pad=18)
    ax.text(0.5, 0.90, terulet, ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc="center", bbox=[0.02, 0.48, 0.96, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f3f3f3")
        if row in [1, 3]:
            cell.set_facecolor("#eeeeee")
    y = 0.39
    for raw_line in change_names_text(df).splitlines():
        wrapped = textwrap.wrap(raw_line, width=125) or [raw_line]
        for line in wrapped:
            ax.text(0.02, y, line, ha="left", va="top", fontsize=10.5, transform=ax.transAxes)
            y -= 0.055
        y -= 0.012
    fig.tight_layout()
    return fig


def make_level_distribution_fig(df: pd.DataFrame, osztaly: str, terulet: str, angol: bool):
    if df.empty:
        return None
    max_level = int(6 if angol else 7)
    rows = [("2024/2025-ös tanév", "Szint_1"), ("2025/2026-os tanév előzetes eredmény", "Szint_2")]
    fig, ax = plt.subplots(figsize=(14, 6.0))
    y_positions = np.arange(len(rows))
    total = max(len(df), 1)
    present_levels = set()

    for y, (label, col) in zip(y_positions, rows):
        left = 0.0
        dist = level_distribution(df, col, max_level)
        for level in range(0, max_level + 1):
            count = dist[level]
            if count == 0:
                continue
            present_levels.add(level)
            pct = count / total * 100
            color = LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)]
            ax.barh(y, pct, left=left, height=0.46, color=color, edgecolor="white")
            label_text = f"{pct:.1f} %".replace(".", ",")
            center = left + pct / 2
            if pct >= 7:
                ax.text(center, y, label_text, ha="center", va="center", fontsize=10, fontweight="bold")
            elif pct >= 3:
                ax.text(center, y, label_text, ha="center", va="center", fontsize=8.5, fontweight="bold")
            else:
                offset_y = -0.34 if y == 0 else 0.34
                label_x = min(max(center, 2.0), 98.0)
                ax.annotate(
                    label_text,
                    xy=(center, y),
                    xytext=(label_x, y + offset_y),
                    textcoords="data",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "black", "shrinkA": 0, "shrinkB": 0},
                    clip_on=False,
                )
            left += pct

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tanulók aránya")
    ax.set_title(f"{osztaly} – {terulet} – Mérési szintek megoszlása", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{x},0 %" for x in [0, 20, 40, 60, 80, 100]])
    legend_levels = [level for level in range(0, max_level + 1) if level in present_levels]
    handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[min(level, len(LEVEL_COLORS)-1)], lw=8,
                   label=("Nincs eredmény" if level == 0 else f"{level}. szint"))
        for level in legend_levels
    ]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=min(len(handles), 8))
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.23, bottom=0.25, right=0.98, top=0.86)
    return fig


def make_level_chart_fig(df: pd.DataFrame, alapszint: int, osztaly: str, terulet: str, angol: bool):
    if df.empty:
        return None
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, df["Szint_1"], width, label="2024/2025")
    ax.bar(x + width / 2, df["Szint_2"], width, label="2025/2026 előzetes")
    max_data_level = int(max(df["Szint_1"].max(), df["Szint_2"].max(), alapszint, 6 if angol else 7))
    ax.set_ylim(0, max_data_level + 0.8)
    if alapszint > 0:
        ax.axhline(alapszint, color="red", linestyle="--", linewidth=2, label=f"Alapszint: {alapszint}", zorder=3)
    ax.set_title(f"{osztaly} – {terulet} – Képességszint változás")
    ax.set_xlabel("Tanuló neve")
    ax.set_ylabel("Képességszint")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Név"], rotation=45, ha="right")
    if angol:
        ticks = sorted(ANGOL_SZINT_LABELS.keys())
        ax.set_yticks(ticks)
        ax.set_yticklabels([ANGOL_SZINT_LABELS[y] for y in ticks])
    else:
        ax.set_yticks(np.arange(0, max_data_level + 1, 1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.08, bottom=0.28, right=0.98, top=0.90)
    return fig

def generate_pdf(df: pd.DataFrame, alapszint: int, osztaly: str, terulet: str, angol: bool, include_summary: bool = True, include_distribution: bool = True) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        add_change_chart(pdf, df, osztaly, terulet)
        if include_summary:
            add_change_summary_table(pdf, df, osztaly, terulet)
        if include_distribution:
            add_level_distribution_chart(pdf, df, osztaly, terulet, angol)
        add_level_chart(pdf, df, alapszint, osztaly, terulet, angol)
    buffer.seek(0)
    return buffer.getvalue()


st.title("Kompetenciamérés PDF generátor")
st.write("Tölts fel egy Excel fájlt, válassz mérési területet és osztályt, majd töltsd le a PDF-et.")

uploaded_file = st.file_uploader("Excel fájl feltöltése", type=["xlsx"])
if uploaded_file is None:
    st.info("Kezdéshez tölts fel egy .xlsx fájlt.")
    st.stop()

sheet_names = get_sheet_names(uploaded_file)
with st.sidebar:
    st.header("Beállítások")
    sheet_name = st.selectbox("Munkalap", sheet_names, index=sheet_names.index("Munka1") if "Munka1" in sheet_names else 0)
    header_row = st.number_input("Fejléc sor száma az Excelben", min_value=1, max_value=20, value=3)

try:
    area_map = detect_areas(uploaded_file, sheet_name)
    area_options = list(area_map.keys()) or DEFAULT_AREAS
except Exception as exc:
    st.error(f"Nem sikerült a mérési területek felismerése: {exc}")
    st.stop()

with st.sidebar:
    terulet = st.selectbox("Kompetenciamérési terület", area_options)

try:
    raw_df = read_selected_area(uploaded_file, sheet_name, terulet, int(header_row) - 1)
except Exception as exc:
    st.error(str(exc))
    st.stop()

class_options = get_classes(raw_df)
with st.sidebar:
    selected_class = st.selectbox("Osztály / tanulócsoport", class_options)
    default_base = 2 if "angol" not in terulet.lower() and "német" not in terulet.lower() else 3
    alapszint = st.number_input("Alapszint (0 = ne jelenjen meg)", min_value=0, max_value=10, value=default_base)
    auto_language = "angol" in terulet.lower() or "német" in terulet.lower()
    angol = st.checkbox("Nyelvi szintek használata (pre-A1, A1, A2...)", value=auto_language)
    include_summary = st.checkbox("3. ábra: változás mértéke táblázat", value=True)
    include_distribution = st.checkbox("4. ábra: mérési szintek halmozott sávdiagram", value=True)
    safe_class = selected_class.replace(" ", "_").replace("/", "-")
    pdf_nev = st.text_input("PDF fájlnév", value=f"{terulet}_{safe_class}.pdf")

try:
    df = prepare_dataframe(raw_df, selected_class, angol)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("A kiválasztott beállításokkal nincs megjeleníthető tanuló.")
    st.stop()

st.subheader("Beolvasott adatok előnézete")
preview_cols = [c for c in ["Évfolyam", "Tanulócsoportok", "Mérési azonosító", "Név", "Képességpont_1", "Képességszint", "Képességpont_2", "Képességszint.1", "Képességpont változás", "Változás kategória", "Szint_1", "Szint_2"] if c in df.columns]
st.dataframe(df[preview_cols], use_container_width=True)
st.caption(f"{len(df)} tanuló • terület: {terulet} • osztály/tanulócsoport: {selected_class}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Változás mértéke összesítő")
    st.dataframe(change_summary(df, selected_class), use_container_width=True, hide_index=True)
with col2:
    st.subheader("Szintmegoszlás")
    max_level_preview = int(max(6 if angol else 7, df["Szint_1"].max(), df["Szint_2"].max()))
    level_rows = []
    for label, col in [("2024/2025", "Szint_1"), ("2025/2026 előzetes", "Szint_2")]:
        dist = level_distribution(df, col, max_level_preview)
        total = max(len(df), 1)
        row = {"Tanév": label}
        for level, count in dist.items():
            if count:
                row[f"{level}. szint"] = f"{count} fő ({count/total*100:.1f}%)".replace(".", ",")
        level_rows.append(row)
    st.dataframe(pd.DataFrame(level_rows).fillna("-"), use_container_width=True, hide_index=True)

st.subheader("Név szerinti bontás a változás mértéke szerint")
st.write("Ez a rész a PDF-be is bekerül, és innen a Word-jelentésbe is bemásolható.")
st.text_area("Másolható szöveg", value=change_names_text(df), height=190)

st.subheader("Diagram előnézetek a PDF jelentésből")
st.write("Az alábbi ábrák ugyanazok, amelyek a PDF jelentésbe is bekerülnek.")

tab1, tab2, tab3, tab4 = st.tabs(["1. ábra", "2. ábra", "3. ábra", "4. ábra"])
with tab1:
    fig = make_change_chart_fig(df, selected_class, terulet)
    if fig is None:
        st.info("Ehhez az ábrához nincs megjeleníthető képességpont-változás.")
    else:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
with tab2:
    fig = make_level_chart_fig(df, int(alapszint), selected_class, terulet, angol)
    if fig is None:
        st.info("Nincs megjeleníthető képességszint-adat.")
    else:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
with tab3:
    if include_summary:
        fig = make_change_summary_fig(df, selected_class, terulet)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("A 3. ábra jelenleg ki van kapcsolva az oldalsávon.")
with tab4:
    if include_distribution:
        fig = make_level_distribution_fig(df, selected_class, terulet, angol)
        if fig is None:
            st.info("Nincs megjeleníthető szintmegoszlás.")
        else:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("A 4. ábra jelenleg ki van kapcsolva az oldalsávon.")

st.subheader("PDF jelentés letöltése")
pdf_bytes = generate_pdf(df, int(alapszint), selected_class, terulet, angol, include_summary, include_distribution)
st.download_button(
    "📄 PDF jelentés letöltése",
    data=pdf_bytes,
    file_name=pdf_nev if pdf_nev.lower().endswith(".pdf") else f"{pdf_nev}.pdf",
    mime="application/pdf",
)
