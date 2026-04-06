import time as pytime
from datetime import date, datetime, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="GLOS - Tipificación diaria",
    page_icon="📞",
    layout="wide",
)

st.markdown(
    """
    <style>
    .pbix-band {
        background: #0b6a8f;
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        font-size: 1.05rem;
        border-radius: 0;
        padding: 0.55rem 0.8rem;
        margin: 0.4rem 0 0.8rem 0;
        letter-spacing: 0.01em;
    }

    .pbix-legend {
        color: rgba(255,255,255,0.84);
        text-align: center;
        font-size: 0.95rem;
        margin: 0.15rem 0 1.15rem 0;
        line-height: 1.35;
    }

    .pbix-center-page-title {
        color: #52b7ea;
        font-size: 1.9rem;
        font-style: italic;
        font-weight: 500;
        margin: 1.1rem 0 0.55rem 0;
        line-height: 1.1;
    }

    .mini-band {
        background: #0b6a8f;
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        font-size: 0.98rem;
        padding: 0.45rem 0.8rem;
        margin: 0.3rem 0 0.85rem 0;
    }

    .metric-pack-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.15rem 0 0.55rem 0;
        color: #f5f7fb;
    }

    .metric-pack-scope {
        color: rgba(255,255,255,0.62);
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 0.90rem;
        line-height: 1.35;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(22,27,34,0.96), rgba(13,17,23,0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid var(--accent, #ff4b4b);
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 110px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        margin-bottom: 12px;
    }

    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.32);
        transition: 0.18s ease;
    }

    .metric-label {
        font-size: 0.92rem;
        font-weight: 600;
        color: rgba(255,255,255,0.78);
        margin-bottom: 10px;
        line-height: 1.25;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.05;
        letter-spacing: -0.02em;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f5f7fb;
        margin: 1.0rem 0 0.2rem 0;
        line-height: 1.1;
    }

    .section-subtitle {
        color: rgba(255,255,255,0.82);
        font-size: 0.96rem;
        margin-bottom: 0.8rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# CONSTANTS
# -------------------------------
API_URL = "https://eva.bonsaif.com/api"
CDMX_TZ = ZoneInfo("America/Mexico_City")

TARGET_COLS = [
    "ID_CC", "Campaña_CC", "Cliente_CC", "Tel_Marcado_CC", "Carrier_CC", "Tipo_Tel_CC",
    "Duracion_CC", "Duracion_Min_CC", "Estatus_CC", "Codigo_Accion_CC", "Codigo_Resultado_CC",
    "Fecha_CC", "Codigo_sip_CC", "Descripcion_sip_CC", "Grabacion_CC", "Extension_CC", "Gestor_CC",
    "Obs_CC", "Origen_CC", "Colgo_Agente_CC", "Salida_CC", "Campo_Clave", "acw",
    "Calificacion_Int_CC", "Sistema"
]

TIP_ORDER_3 = ["CONTACTO", "IMPROCEDENTE", "NO CONTACTADO", "OTROS / SIN CALIFICACION"]
TIP_ABBR = {
    "CONTACTO": "CTO",
    "IMPROCEDENTE": "IMP",
    "NO CONTACTADO": "NCT",
    "OTROS / SIN CALIFICACION": "OTR"
}


# -------------------------------
# SECRETS
# -------------------------------
def get_secret(section: str, key: str, default: str = "") -> str:
    try:
        return str(st.secrets[section][key]).strip()
    except Exception:
        return default


GLOS_API_KEY = get_secret(
    "bonsaif",
    "glos_api_key",
    get_secret("bonsaif", "api_key")
)
GLOS_SYS = get_secret("bonsaif", "glos_sys", "cc61")


# -------------------------------
# HELPERS
# -------------------------------
def mexico_now() -> datetime:
    return datetime.now(CDMX_TZ)


def to_clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_status(x) -> str:
    t = to_clean_text(x).upper()
    return "SIN CALIFICACION" if t == "" else t


def status_group_3(status: str) -> str:
    s = normalize_status(status)
    if s in {"CONTACTO", "IMPROCEDENTE", "NO CONTACTADO"}:
        return s
    return "OTROS / SIN CALIFICACION"


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def choose_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    fallback_name: str,
    fallback_value: str
) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    for c in candidates:
        if c in out.columns:
            return out, c

    out[fallback_name] = fallback_value
    return out, fallback_name


def compute_business_reference_day(today: date) -> date:
    ayer1 = today - timedelta(days=1)

    is_dec26 = today.month == 12 and today.day == 26
    is_jan2 = today.month == 1 and today.day == 2
    is_feb2 = today.month == 2 and today.day == 2
    is_feb3 = today.month == 2 and today.day == 3
    is_mar16 = today.month == 3 and today.day == 16
    is_mar17 = today.month == 3 and today.day == 17
    is_apr6_2026 = today == date(2026, 4, 6)

    if is_dec26:
        return date(today.year, 12, 24)
    elif is_jan2:
        return date(today.year - 1, 12, 31)
    elif is_feb3:
        return today - timedelta(days=3)
    elif is_feb2:
        return today - timedelta(days=1)
    elif is_mar16:
        return today - timedelta(days=2)
    elif is_mar17:
        return today - timedelta(days=3)
    elif is_apr6_2026:
        return date(2026, 4, 2)
    elif ayer1.weekday() == 6:
        return today - timedelta(days=2)
    else:
        return ayer1

def clean_name(t):
    if pd.isna(t):
        return None

    s = str(t).strip().upper()
    s = (
        s.replace("Á", "A")
         .replace("É", "E")
         .replace("Í", "I")
         .replace("Ó", "O")
         .replace("Ú", "U")
         .replace("Ñ", "N")
         .replace(".", " ")
         .replace(",", " ")
         .replace("/", " ")
         .replace("-", " ")
    )
    s = " ".join([p for p in s.split(" ") if p != ""])
    return s


def build_detail_rename_map(team_col: str, agent_col: str) -> dict:
    rename_map = {
        "Sistema": "Sistema",
        team_col: "Equipo",
        agent_col: "Agente",
        "Tipificacion_3": "Tipificación general",
        "Tipificacion_Detalle": "Tipificación detalle",
        "Tel_Marcado_CC": "Teléfono",
        "Campaña_CC": "Campaña",
        "Cliente_CC": "Cliente",
        "Duracion_CC": "Duración (seg)",
        "Duracion_Min_CC": "Duración (min)",
        "Codigo_Accion_CC": "Código acción",
        "Codigo_Resultado_CC": "Código resultado",
        "Extension_CC": "Extensión",
        "Descripcion_sip_CC": "Descripción SIP",
        "Campo_Clave": "Campo clave",
        "Grabacion_CC": "Grabación",
        "Fecha_CC": "Fecha",
    }

    if team_col != "Calificacion_Int_CC":
        rename_map["Calificacion_Int_CC"] = "Equipo original"

    if agent_col != "Extension_CC":
        rename_map["Extension_CC"] = "Extensión"

    return rename_map


# -------------------------------
# API FETCHERS
# -------------------------------
def fetch_api_records(params: dict) -> list[dict]:
    response = requests.get(API_URL, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result", [])
    if isinstance(result, list):
        return result
    return []


def normalize_api_df(records: list[dict]) -> pd.DataFrame:
    cols = [
        "ID_CC", "Campaña_CC", "Cliente_CC", "Tel_Marcado_CC", "Carrier_CC", "Tipo_Tel_CC",
        "Duracion_CC", "Duracion_Min_CC", "Estatus_CC", "Codigo_Accion_CC", "Codigo_Resultado_CC",
        "Fecha_CC", "Codigo_sip_CC", "Descripcion_sip_CC", "Grabacion_CC", "Extension_CC", "Gestor_CC",
        "Obs_CC", "Origen_CC", "Colgo_Agente_CC", "Salida_CC", "Campo_Clave", "acw",
        "Calificacion_Int_CC", "Clave_int_cli"
    ]

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].copy()

    text_cols = [
        "Campaña_CC", "Cliente_CC", "Tel_Marcado_CC", "Carrier_CC", "Tipo_Tel_CC",
        "Estatus_CC", "Codigo_Accion_CC", "Codigo_Resultado_CC", "Codigo_sip_CC",
        "Descripcion_sip_CC", "Grabacion_CC", "Extension_CC", "Gestor_CC",
        "Obs_CC", "Colgo_Agente_CC", "Salida_CC", "Calificacion_Int_CC", "Clave_int_cli"
    ]
    num_cols = ["ID_CC", "Duracion_CC", "Duracion_Min_CC", "Origen_CC", "Campo_Clave", "acw"]

    for c in text_cols:
        df[c] = df[c].astype("string")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Fecha_CC"] = pd.to_datetime(df["Fecha_CC"], errors="coerce")

    nuevo = df["Clave_int_cli"].fillna("").astype(str).str.strip()
    viejo = df["Calificacion_Int_CC"].fillna("").astype(str).str.strip()

    df["Calificacion_Int_CC"] = np.where(
        nuevo != "",
        nuevo,
        np.where(viejo != "", viejo, np.nan)
    )

    df = df.drop(columns=["Clave_int_cli"])
    df["Sistema"] = "GLOS"

    return ensure_columns(df, TARGET_COLS)


def fetch_glos_raw(pdate: date) -> pd.DataFrame:
    if not GLOS_API_KEY or not GLOS_SYS:
        return pd.DataFrame(columns=TARGET_COLS)

    pdate_text = pdate.strftime("%Y-%m-%d")

    params_primary = {
        "service": "cc/api",
        "m": "27",
        "key": GLOS_API_KEY,
        "sys": GLOS_SYS,
        "fecha_ini": pdate_text,
        "fecha_fin": pdate_text,
    }

    params_fallback = {
        "service": "cc/api",
        "m": "27",
        "key": GLOS_API_KEY,
        "sys": GLOS_SYS,
        "fechaini": pdate_text,
        "fechafin": pdate_text,
    }

    for attempt in range(3):
        try:
            records = fetch_api_records(params_primary)
            if not records:
                records = fetch_api_records(params_fallback)
            return normalize_api_df(records)
        except Exception:
            pytime.sleep(1 + attempt)

    return pd.DataFrame(columns=TARGET_COLS)


# -------------------------------
# SOURCE PREP / EXACT LOGIC
# -------------------------------
def build_glos_exact(raw_df: pd.DataFrame, overwrite_fecha: datetime | None = None) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    out = raw_df.copy()

    out["Estatus_CC"] = out["Estatus_CC"].apply(
        lambda x: "SIN CALIFICACION" if str(x).strip() == "" or pd.isna(x) else x
    )

    if overwrite_fecha is not None:
        out["Fecha_CC"] = pd.Timestamp(overwrite_fecha.replace(tzinfo=None))

    out["_row_order"] = np.arange(len(out))
    out = out.sort_values(
        by=["Tel_Marcado_CC", "Fecha_CC", "_row_order"],
        ascending=[True, False, True],
        kind="mergesort",
        na_position="last"
    )

    out = out.drop_duplicates(subset=["Tel_Marcado_CC"], keep="first").copy()
    out = out.drop(columns=["_row_order"])

    out["Tipificacion_Detalle"] = out["Estatus_CC"].apply(normalize_status)
    out["Tipificacion_3"] = out["Estatus_CC"].apply(status_group_3)
    out["Tipificacion_3_Abbr"] = out["Tipificacion_3"].map(TIP_ABBR).fillna("OTR")

    for c in ["Gestor_CC", "Calificacion_Int_CC", "Extension_CC"]:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str).str.strip()

    return out.reset_index(drop=True)


def apply_powerbi_glos_page_scope(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # Misma estructura lógica que CC2/JV:
    # un bloque para portada y un bloque para la página específica.
    # En GLOS, al ser una sola operación, la página específica usa el
    # universo consolidado de GLOS ya normalizado.
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_consolidado_ayer() -> tuple[pd.DataFrame, date]:
    hoy = mexico_now().date()

    dia_vencido = compute_business_reference_day(hoy)
    target = compute_business_reference_day(dia_vencido)

    raw0 = fetch_glos_raw(target)
    if raw0.empty:
        alt = compute_business_reference_day(target)
        raw0 = fetch_glos_raw(alt)
        target = alt

    combined = build_glos_exact(raw0)
    return combined, target


@st.cache_data(ttl=60, show_spinner=False)
def get_consolidado_hoy() -> tuple[pd.DataFrame, date]:
    now_cdmx = mexico_now()
    hoy = now_cdmx.date()

    day_vencido = compute_business_reference_day(hoy)
    raw = fetch_glos_raw(day_vencido)

    if raw.empty:
        alt = day_vencido - timedelta(days=1)
        raw = fetch_glos_raw(alt)
        day_vencido = alt

    combined = build_glos_exact(raw, overwrite_fecha=now_cdmx)
    return combined, day_vencido


@st.cache_data(ttl=60, show_spinner=False)
def get_consolidado_exact_day(pdate: date) -> pd.DataFrame:
    raw = fetch_glos_raw(pdate)
    return build_glos_exact(raw)


# -------------------------------
# KPI HELPERS
# -------------------------------
def avg_active_seconds_by_tip(df: pd.DataFrame, tip: str) -> float:
    if df.empty or "Duracion_CC" not in df.columns:
        return 0.0

    s = pd.to_numeric(
        df.loc[df["Tipificacion_3"] == tip, "Duracion_CC"],
        errors="coerce"
    ).dropna()

    s = s[s > 0]

    if s.empty:
        return 0.0

    return float(s.mean())


def calc_agent_hangup_pct(df: pd.DataFrame) -> float:
    if df.empty or "Estatus_CC" not in df.columns or "Colgo_Agente_CC" not in df.columns:
        return 0.0

    status = df["Estatus_CC"].fillna("").astype(str).str.strip().str.upper()
    colgo = df["Colgo_Agente_CC"].fillna("").astype(str).str.strip().str.upper()

    llamadas = int(status.eq("CONTACTO").sum())
    if llamadas == 0:
        return 0.0

    colgar = int((status.eq("CONTACTO") & colgo.eq("SI")).sum())
    return float((colgar / llamadas) * 100)


def calc_discard_block_count(df: pd.DataFrame) -> int:
    if df.empty or "Duracion_CC" not in df.columns:
        return 0

    dur = pd.to_numeric(df["Duracion_CC"], errors="coerce")
    return int(dur.eq(0).sum())


def calc_awc(df: pd.DataFrame) -> int:
    if df.empty or "acw" not in df.columns:
        return 0

    s = pd.to_numeric(df["acw"], errors="coerce").dropna()
    if s.empty:
        return 0

    return int(round(s.mean()))


def compute_metric_pack(df: pd.DataFrame) -> dict:
    return {
        "Contacto (avg sec)": avg_active_seconds_by_tip(df, "CONTACTO"),
        "Improcedente (avg sec)": avg_active_seconds_by_tip(df, "IMPROCEDENTE"),
        "No contactado (avg sec)": avg_active_seconds_by_tip(df, "NO CONTACTADO"),
        "% Colgó Agente": calc_agent_hangup_pct(df),
        "Bloqueo de Discard": calc_discard_block_count(df),
        "AWC": calc_awc(df),
    }


def fmt_metric_value(label: str, value) -> str:
    if "%" in label:
        return f"{float(value):.0f}%"

    if (
        "Bloqueo" in label
        or label == "AWC"
        or "Registros" in label
        or "Total" in label
        or "Otros" in label
    ):
        try:
            return f"{int(round(float(value))):,}"
        except Exception:
            return str(value)

    try:
        return f"{float(value):,.1f}"
    except Exception:
        return str(value)


def render_pbix_band(title: str):
    st.markdown(f'<div class="pbix-band">{title}</div>', unsafe_allow_html=True)


def render_pbix_legend():
    st.markdown(
        '<div class="pbix-legend">CTO = CONTACTO &nbsp;&nbsp;&nbsp; IMP = IMPROCEDENTE &nbsp;&nbsp;&nbsp; NCT = NO CONTACTADO &nbsp;&nbsp;&nbsp; AWC = TIEMPO P/ TIPIFICAR</div>',
        unsafe_allow_html=True
    )


def render_center_page_title(title: str):
    st.markdown(f'<div class="pbix-center-page-title">{title}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value, accent: str = "#ff4b4b", display_label: str | None = None):
    label_show = display_label if display_label is not None else label

    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}">
            <div class="metric-label">{label_show}</div>
            <div class="metric-value">{fmt_metric_value(label, value)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_pack(
    title: str,
    pack: dict,
    accent: str = "#ff4b4b",
    subtitle: str = "",
    row1_header: str = "",
    row2_header: str = ""
):
    st.markdown(f'<div class="metric-pack-title">{title}</div>', unsafe_allow_html=True)

    if subtitle:
        st.markdown(f'<div class="metric-pack-scope">{subtitle}</div>', unsafe_allow_html=True)

    if row1_header:
        st.markdown(f'<div class="mini-band">{row1_header}</div>', unsafe_allow_html=True)

    row1 = st.columns(3)
    with row1[0]:
        render_metric_card(
            "Contacto (avg sec)",
            pack["Contacto (avg sec)"],
            accent,
            "Tiempo Promedio de Segundos<br>Activos<br>Contacto"
        )
    with row1[1]:
        render_metric_card(
            "Improcedente (avg sec)",
            pack["Improcedente (avg sec)"],
            accent,
            "Tiempo Promedio de Segundos<br>Activos<br>Improcedente"
        )
    with row1[2]:
        render_metric_card(
            "No contactado (avg sec)",
            pack["No contactado (avg sec)"],
            accent,
            "Tiempo Promedio de Segundos<br>Activos<br>No contactado"
        )

    if row2_header:
        st.markdown(f'<div class="mini-band">{row2_header}</div>', unsafe_allow_html=True)

    row2 = st.columns(3)
    with row2[0]:
        render_metric_card(
            "% Colgó Agente",
            pack["% Colgó Agente"],
            accent,
            "Colgó Agente<br>En porcentaje"
        )
    with row2[1]:
        render_metric_card(
            "Bloqueo de Discard",
            pack["Bloqueo de Discard"],
            accent,
            "Bloqueo de Discard<br>No. de llamadas"
        )
    with row2[2]:
        render_metric_card(
            "AWC",
            pack["AWC"],
            accent,
            "AWC<br>Segundos"
        )


# -------------------------------
# SUMMARY / EXPORT HELPERS
# -------------------------------
def build_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "Registros", "Porcentaje"])

    out = (
        df.groupby(group_col, dropna=False)
        .size()
        .reset_index(name="Registros")
        .sort_values("Registros", ascending=False)
    )

    total = out["Registros"].sum()
    out["Porcentaje"] = np.where(total > 0, out["Registros"] / total, 0.0)
    return out


def build_team_agent_summary(df: pd.DataFrame, tip_col: str, team_col: str, agent_col: str):
    work_local = df.copy()

    if team_col not in work_local.columns:
        work_local[team_col] = "Sin equipo"
    if agent_col not in work_local.columns:
        work_local[agent_col] = "Sin agente"

    work_local[team_col] = work_local[team_col].replace("", np.nan).fillna("Sin equipo")
    work_local[agent_col] = work_local[agent_col].replace("", np.nan).fillna("Sin agente")

    team = (
        work_local.groupby([team_col, tip_col], dropna=False)
        .size()
        .reset_index(name="Registros")
    )

    agent = (
        work_local.groupby([agent_col, tip_col], dropna=False)
        .size()
        .reset_index(name="Registros")
    )

    return team, agent


def build_pbix_agent_table(df: pd.DataFrame, team_col: str, agent_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            team_col, agent_col,
            "Duración Promedio Contacto", "Total Contacto",
            "Duración Promedio Improcedente", "Total Improcedente",
            "Duración Promedio No Contactado", "Total No Contactado",
            "AWC"
        ])

    work_local = df.copy()
    work_local["Duracion_CC"] = pd.to_numeric(work_local["Duracion_CC"], errors="coerce")
    work_local["acw"] = pd.to_numeric(work_local["acw"], errors="coerce")
    work_local["Estatus_CC"] = work_local["Estatus_CC"].fillna("").astype(str).str.strip().str.upper()

    rows = []
    for (equipo, agente), g in work_local.groupby([team_col, agent_col], dropna=False):
        g_cto = g[g["Estatus_CC"] == "CONTACTO"]
        g_imp = g[g["Estatus_CC"] == "IMPROCEDENTE"]
        g_nct = g[g["Estatus_CC"] == "NO CONTACTADO"]

        cto_secs = g_cto["Duracion_CC"].dropna()
        imp_secs = g_imp["Duracion_CC"].dropna()
        nct_secs = g_nct["Duracion_CC"].dropna()
        awc_vals = g["acw"].dropna()

        cto_secs = cto_secs[cto_secs > 0]
        imp_secs = imp_secs[imp_secs > 0]
        nct_secs = nct_secs[nct_secs > 0]

        rows.append({
            team_col: equipo,
            agent_col: agente,
            "Duración Promedio Contacto": float(cto_secs.mean()) if not cto_secs.empty else 0.0,
            "Total Contacto": int(len(g_cto)),
            "Duración Promedio Improcedente": float(imp_secs.mean()) if not imp_secs.empty else 0.0,
            "Total Improcedente": int(len(g_imp)),
            "Duración Promedio No Contactado": float(nct_secs.mean()) if not nct_secs.empty else 0.0,
            "Total No Contactado": int(len(g_nct)),
            "AWC": float(awc_vals.mean()) if not awc_vals.empty else 0.0,
        })

    return pd.DataFrame(rows).sort_values([team_col, agent_col]).reset_index(drop=True)


def make_excel(
    detail_df: pd.DataFrame,
    summary_tip: pd.DataFrame,
    summary_team: pd.DataFrame,
    summary_agent: pd.DataFrame
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_tip.to_excel(writer, sheet_name="Resumen_Tipificacion", index=False)
        summary_team.to_excel(writer, sheet_name="Resumen_Equipo", index=False)
        summary_agent.to_excel(writer, sheet_name="Resumen_Agente", index=False)
        detail_df.to_excel(writer, sheet_name="Detalle", index=False)
    buffer.seek(0)
    return buffer.getvalue()


# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.subheader("Conexión GLOS")

    if not GLOS_API_KEY or not GLOS_SYS:
        st.error("Faltan credenciales API en .streamlit/secrets.toml")

    if st.button("Recargar datos ahora"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Período")
    view_mode = st.radio(
        "Selecciona el período",
        ["Hoy", "Día hábil anterior", "Elegir fecha"],
        index=1
    )

    selected_date = None
    if view_mode == "Elegir fecha":
        selected_date = st.date_input("Fecha", value=mexico_now().date())

    st.subheader("Vista de tipificación")
    tip_view = st.radio(
        "Nivel de detalle",
        ["Resumen general", "Detalle completo"],
        index=1
    )


# -------------------------------
# VALIDATION
# -------------------------------
if not GLOS_API_KEY or not GLOS_SYS:
    st.title("Tipificaciones de Contacto - GLOS")
    st.warning("Agrega tus credenciales API en `.streamlit/secrets.toml`.")
    st.stop()


# -------------------------------
# TITLE
# -------------------------------
st.title("Tipificaciones de Contacto - GLOS")
st.caption("Resumen general y desglose de la operación GLOS.")


# -------------------------------
# LOAD DATA
# -------------------------------
if view_mode == "Hoy":
    df, source_date = get_consolidado_hoy()
    visual_label = f"Hoy ({mexico_now().strftime('%Y-%m-%d %H:%M:%S')})"
    source_label = f"Datos reales de: {source_date}"
elif view_mode == "Día hábil anterior":
    df, source_date = get_consolidado_ayer()
    visual_label = f"Día hábil anterior ({source_date})"
    source_label = f"Datos reales de: {source_date}"
else:
    df = get_consolidado_exact_day(selected_date)
    source_date = selected_date
    visual_label = f"Fecha elegida ({selected_date})"
    source_label = f"Datos reales de: {source_date}"

if df.empty:
    st.error("No se encontraron registros para la vista seleccionada.")
    st.stop()


# -------------------------------
# FILTERS
# -------------------------------
work = df.copy()

work, team_col = choose_existing_column(
    work,
    ["Calificacion_Int_CC", "Supervisor_Excel", "Equipo", "Supervisor"],
    "Equipo",
    "Sin equipo"
)

work, agent_col = choose_existing_column(
    work,
    ["Gestor_CC", "Agente", "Nombre_Agente", "Extension_CC"],
    "Agente",
    "Sin agente"
)

work[team_col] = work[team_col].replace("", np.nan).fillna("Sin equipo").astype(str)
work[agent_col] = work[agent_col].replace("", np.nan).fillna("Sin agente").astype(str)

tip_col = "Tipificacion_3" if tip_view == "Resumen general" else "Tipificacion_Detalle"

with st.sidebar:
    st.markdown("---")
    st.subheader("Filtros")

    team_options = sorted(work[team_col].dropna().astype(str).unique().tolist())
    selected_teams = st.multiselect("Equipo", team_options, default=team_options)

    work_team = work[work[team_col].isin(selected_teams)].copy() if selected_teams else work.iloc[0:0].copy()

    agent_options = sorted(work_team[agent_col].dropna().astype(str).unique().tolist())
    selected_agents = st.multiselect("Agente", agent_options, default=agent_options)

    work_agent = work_team[work_team[agent_col].isin(selected_agents)].copy() if selected_agents else work_team.iloc[0:0].copy()

    tip_options = sorted(work_agent[tip_col].dropna().astype(str).unique().tolist())
    selected_tip = st.multiselect("Tipificación", tip_options, default=tip_options)

if selected_teams:
    work = work[work[team_col].isin(selected_teams)].copy()
else:
    work = work.iloc[0:0].copy()

if selected_agents:
    work = work[work[agent_col].isin(selected_agents)].copy()
else:
    work = work.iloc[0:0].copy()

if selected_tip:
    work = work[work[tip_col].isin(selected_tip)].copy()
else:
    work = work.iloc[0:0].copy()

if work.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

TEAM_LABEL = "Equipo"
AGENT_LABEL = "Agente"
chart_work = work.copy()


# -------------------------------
# KPIS
# -------------------------------
page1_base = df.copy()
page1_glos_pack = compute_metric_pack(page1_base)

page2_base = apply_powerbi_glos_page_scope(df.copy())
page2_glos_pack = compute_metric_pack(page2_base)

st.caption(f"Vista: {visual_label} | {source_label}")
st.caption(f"Al corte del {mexico_now().strftime('%m/%d/%Y %I:%M:%S %p')}")

render_pbix_band("MÉTRICAS GLOBALES")
render_pbix_legend()

render_metric_pack(
    "GLOS",
    page1_glos_pack,
    accent="#52b7ea"
)

render_center_page_title("GLOS")
render_metric_pack(
    "Indicadores de la operación",
    page2_glos_pack,
    accent="#52b7ea",
    row1_header="Tiempo Promedio de Segundos Activos",
    row2_header="% Colgó Agente / Bloqueo de Discard / AWC"
)


# -------------------------------
# SUMMARIES
# -------------------------------
summary_tip = build_summary(chart_work, tip_col)

if tip_col == "Tipificacion_3":
    order_map = {k: i for i, k in enumerate(TIP_ORDER_3)}
    summary_tip["OrdenTmp"] = summary_tip[tip_col].map(order_map).fillna(999)
    summary_tip = summary_tip.sort_values(["OrdenTmp", "Registros"], ascending=[True, False]).drop(columns="OrdenTmp")

team_summary_long, agent_summary_long = build_team_agent_summary(chart_work, tip_col, team_col, agent_col)

team_pivot = (
    team_summary_long.pivot(index=team_col, columns=tip_col, values="Registros")
    .fillna(0)
    .reset_index()
)

agent_pivot = build_pbix_agent_table(work, team_col, agent_col)


# -------------------------------
# CHARTS
# -------------------------------
st.markdown('<div class="section-title">Desglose por tipificación</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Distribución de registros en la vista consultada.</div>', unsafe_allow_html=True)

left, right = st.columns([0.95, 1.05])

with left:
    fig_tip = px.bar(
        summary_tip,
        x=tip_col,
        y="Registros",
        text="Registros",
        title="Registros por tipificación",
    )
    fig_tip.update_traces(textposition="outside")
    fig_tip.update_layout(
        xaxis_title="Tipificación",
        yaxis_title="Registros",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig_tip, use_container_width=True)

with right:
    fig_donut = px.pie(
        summary_tip,
        names=tip_col,
        values="Registros",
        hole=0.55,
        title="Participación porcentual por tipificación",
    )
    fig_donut.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown('<div class="section-title">Desglose por equipo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Volumen de registros por equipo y tipificación.</div>', unsafe_allow_html=True)

team_chart_df = team_summary_long.sort_values(
    [team_col, "Registros"],
    ascending=[True, False]
).copy()

team_chart_df["Label"] = team_chart_df["Registros"].astype(int).astype(str)

fig_team = px.bar(
    team_chart_df,
    x=team_col,
    y="Registros",
    color=tip_col,
    text="Label",
    title="Volumen de registros por equipo",
    barmode="stack",
)

fig_team.update_traces(
    textposition="inside",
    textfont_size=11,
    insidetextanchor="middle",
    cliponaxis=False
)

fig_team.update_layout(
    xaxis_title=TEAM_LABEL,
    yaxis_title="Registros",
    uniformtext_minsize=8,
    uniformtext_mode="hide"
)

st.plotly_chart(fig_team, use_container_width=True)

st.markdown('<div class="section-title">Desglose por agente</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Volumen de registros por agente y tipificación.</div>', unsafe_allow_html=True)

agent_chart_df = agent_summary_long.sort_values(
    [agent_col, "Registros"],
    ascending=[True, False]
).copy()

agent_chart_df["Label"] = np.where(
    agent_chart_df["Registros"] >= 15,
    agent_chart_df["Registros"].astype(int).astype(str),
    ""
)

fig_agent = px.bar(
    agent_chart_df,
    x=agent_col,
    y="Registros",
    color=tip_col,
    text="Label",
    title="Volumen de registros por agente",
    barmode="stack",
)

fig_agent.update_traces(
    textposition="inside",
    textfont_size=10,
    insidetextanchor="middle",
    cliponaxis=False
)

fig_agent.update_layout(
    xaxis_title=AGENT_LABEL,
    yaxis_title="Registros",
    uniformtext_minsize=8,
    uniformtext_mode="hide"
)

st.plotly_chart(fig_agent, use_container_width=True)


# -------------------------------
# TABLES
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen por tipificación",
    "Concentrado por equipo",
    "Concentrado por agente",
    "Detalle de llamadas"
])

with tab1:
    show = summary_tip.copy()
    show["Porcentaje"] = (show["Porcentaje"] * 100).round(2)
    show = show.rename(columns={tip_col: "Tipificación"})
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    team_show = team_pivot.copy().rename(columns={team_col: TEAM_LABEL})
    st.dataframe(team_show, use_container_width=True, hide_index=True)

with tab3:
    agent_show = agent_pivot.copy().rename(columns={
        team_col: TEAM_LABEL,
        agent_col: AGENT_LABEL
    })
    st.dataframe(agent_show, use_container_width=True, hide_index=True)

with tab4:
    detail_cols = [
        "Fecha_CC", "Sistema", team_col, agent_col, "Tipificacion_3", "Tipificacion_Detalle",
        "Tel_Marcado_CC", "Campaña_CC", "Cliente_CC", "Duracion_CC", "Duracion_Min_CC",
        "Codigo_Accion_CC", "Codigo_Resultado_CC", "Extension_CC", "Calificacion_Int_CC",
        "Descripcion_sip_CC", "Campo_Clave", "Grabacion_CC"
    ]

    detail_cols = [c for c in detail_cols if c in work.columns]
    detail_cols = list(dict.fromkeys(detail_cols))

    detail_df = work[detail_cols].sort_values("Fecha_CC", ascending=False).copy()
    detail_df = detail_df.rename(columns=build_detail_rename_map(team_col, agent_col))

    st.dataframe(detail_df, use_container_width=True, hide_index=True)


# -------------------------------
# DOWNLOAD
# -------------------------------
excel_detail = work.sort_values("Fecha_CC", ascending=False).copy()
excel_detail = excel_detail.rename(columns=build_detail_rename_map(team_col, agent_col))

excel_summary_tip = summary_tip.copy().rename(columns={tip_col: "Tipificación"})
excel_team = team_pivot.copy().rename(columns={team_col: TEAM_LABEL})
excel_agent = agent_pivot.copy().rename(columns={
    team_col: TEAM_LABEL,
    agent_col: AGENT_LABEL
})

excel_bytes = make_excel(
    detail_df=excel_detail,
    summary_tip=excel_summary_tip,
    summary_team=excel_team,
    summary_agent=excel_agent
)

st.download_button(
    "Descargar Excel",
    data=excel_bytes,
    file_name=f"glos_tipificacion_{source_date}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
