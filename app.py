import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Kompetenciamérés PDF generátor", layout="wide")

ANGOL_SZINT_MAP = {
    "pre-a1": 1,
    "a1": 2,
    "a2": 3,
    "b1": 4,
    "b2": 5,
    "c1": 6,
}

ANGOL_SZINT_LABELS = {
    0: "0. szint - pre-A1 alatti szint",
    1: "1. szint - pre-A1 szint",
    2: "2. szint - A1 szint",
    3: "3. szint - A2 szint",
    4: "4. szint - B1 szint",
    5: "5. szint - B2 szint",
    6: "6. szint - C1 szint vagy fölötte",
}

REQUIRED_COLUMNS = [
    "Név",
    "Képességszint",
    "Képességszint.1",
    "Képességpont változás",
]


def normalize_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """A mintafájlban NEVEK szerepel, a régi kód Név oszlopot vár."""
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


def load_excel(uploaded_file, sheet_name: str, header_row_index: int) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row_index)
    df = normalize_name_columns(df)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Hiányzó oszlop(ok): " + ", ".join(missing) +
            ". Ellenőrizd a fejlécsort és a munkalapot."
        )
    return df


def prepare_dataframe(df: pd.DataFrame, angol: bool) -> pd.DataFrame:
    df = df.copy()
    if angol:
        df["Szint_1"] = df["Képességszint"].apply(decode_angol_szint)
        df["Szint_2"] = df["Képességszint.1"].apply(decode_angol_szint)
    else:
        df["Szint_1"] = df["Képességszint"].apply(decode_numeric_level)
        df["Szint_2"] = df["Képességszint.1"].apply(decode_numeric_level)

    df["Képességpont változás"] = pd.to_numeric(
        df["Képességpont változás"], errors="coerce"
    ).fillna(0)
    df["Név"] = df["Név"].astype(str).apply(lambda x: " ".join(x.split()))
    df = df[df["Név"].str.lower().ne("nan")]
    df = df.sort_values(by="Képességpont változás")
    return df


def add_change_chart(pdf: PdfPages, df: pd.DataFrame, osztaly: str, terulet: str) -> None:
    if df["Képességpont változás"].abs().sum() <= 0:
        return

    colors, _ = zip(*[categorize_change(v) for v in df["Képességpont változás"]])

    plt.figure(figsize=(14, 6))
    bars = plt.bar(df["Név"], df["Képességpont változás"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    plt.title(f"{osztaly} – {terulet} – Képességpont Változás")
    plt.xlabel("Tanuló neve")
    plt.ylabel("Képességpont változás")
    plt.xticks(rotation=45, ha="right")

    legend_labels = {
        "Jelentős romlás": "#8B0000",
        "Mérsékelt romlás": "#FF6347",
        "Elhanyagolható romlás/javulás": "gray",
        "Mérsékelt javulás": "#90EE90",
        "Jelentős javulás": "#006400",
    }
    patches = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in legend_labels.items()
    ]
    plt.legend(handles=patches, title="Kategóriák")
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def add_level_chart(pdf: PdfPages, df: pd.DataFrame, alapszint: int, osztaly: str, terulet: str, angol: bool) -> None:
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, df["Szint_1"], width, label="2023/2024")
    plt.bar(x + width / 2, df["Szint_2"], width, label="2024/2025 (előzetes)")

    if alapszint > 0:
        plt.axhline(alapszint, color="red", linestyle="--", linewidth=2, label=f"Alapszint: {alapszint}")

    plt.title(f"{osztaly} – {terulet} – Képességszint Változás")
    plt.xlabel("Tanuló neve")
    plt.ylabel("Képességszint")
    plt.xticks(x, df["Név"], rotation=45, ha="right")

    if angol:
        yticks = sorted(ANGOL_SZINT_LABELS.keys())
        ylabels = [ANGOL_SZINT_LABELS[y] for y in yticks]
        plt.yticks(yticks, ylabels)
    else:
        plt.yticks(np.arange(0, 8, 1))

    plt.legend()
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def generate_pdf(df: pd.DataFrame, alapszint: int, osztaly: str, terulet: str, angol: bool) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        add_change_chart(pdf, df, osztaly, terulet)
        add_level_chart(pdf, df, alapszint, osztaly, terulet, angol)
    buffer.seek(0)
    return buffer.getvalue()


st.title("Kompetenciamérés PDF generátor")
st.write("Tölts fel egy Excel fájlt, add meg az adatokat, majd töltsd le a generált PDF-et.")

uploaded_file = st.file_uploader("Excel fájl feltöltése", type=["xlsx"])

with st.sidebar:
    st.header("Beállítások")
    sheet_name = st.text_input("Munkalap neve", value="Munka1")
    header_row = st.number_input("Fejléc sor száma az Excelben", min_value=1, max_value=20, value=3)
    alapszint = st.number_input("Alapszint (0 = ne jelenjen meg)", min_value=0, max_value=10, value=2)
    osztaly = st.text_input("Osztály", value="6.A")
    terulet = st.text_input("Kompetenciamérési terület", value="Szövegértés")
    angol = st.checkbox("Angol kompetenciamérési terület", value=False)
    pdf_nev = st.text_input("PDF fájlnév", value="eredmenyek.pdf")

if uploaded_file is None:
    st.info("Kezdéshez tölts fel egy .xlsx fájlt.")
    st.stop()

try:
    raw_df = load_excel(uploaded_file, sheet_name, int(header_row) - 1)
    df = prepare_dataframe(raw_df, angol)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Beolvasott adatok előnézete")
st.dataframe(df[["Név", "Képességszint", "Képességszint.1", "Képességpont változás", "Szint_1", "Szint_2"]], use_container_width=True)

pdf_bytes = generate_pdf(df, int(alapszint), osztaly, terulet, angol)

st.download_button(
    "PDF letöltése",
    data=pdf_bytes,
    file_name=pdf_nev if pdf_nev.lower().endswith(".pdf") else f"{pdf_nev}.pdf",
    mime="application/pdf",
)
