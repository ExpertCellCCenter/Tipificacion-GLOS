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
def get_bonsaif_secret(section: str, key: str, default: str = "") -> str:
    try:
        return str(st.secrets[section][key]).strip()
    except Exception:
        return default


GLOS_API_KEY = get_bonsaif_secret(
    "bonsaif",
    "glos_api_key",
    get_bonsaif_secret("bonsaif", "api_key")
)
GLOS_SYS = get_bonsaif_secret("bonsaif", "glos_sys", "cc61")


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

    if is_dec26:
        return date(today.year, 12, 24)
    elif is_jan2:
        return date(today.year - 1, 12, 31)
    elif is_feb3:
        return today - timedelta(days=3)
    elif is_feb2:
        return today - timedelta(days=1)
    elif ayer1.weekday() == 6:
        return today - timedelta(days=2)
    else:
        return ayer1


def build_detail_rename_map(team_col: str, agent_col: str) -> dict:
    rename_map = {
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
        "Sistema": "Sistema",
    }

    if team_col != "Calificacion_Int_CC":
        rename_map["Calificacion_Int_CC"] = "Equipo interno"

    if agent_col != "Extension_CC":
        rename_map["Extension_CC"] = "Extensión"

    return rename_map


# -------------------------------
# POWER BI / GLOS EXACT-LIKE BASE
# -------------------------------
def fetch_glos_raw(pdate: date, api_key: str, sys_code: str) -> pd.DataFrame:
    base_cols = [
        "ID_CC", "Campaña_CC", "Cliente_CC", "Tel_Marcado_CC", "Carrier_CC", "Tipo_Tel_CC",
        "Duracion_CC", "Duracion_Min_CC", "Estatus_CC", "Codigo_Accion_CC", "Codigo_Resultado_CC",
        "Fecha_CC", "Codigo_sip_CC", "Descripcion_sip_CC", "Grabacion_CC", "Extension_CC", "Gestor_CC",
        "Obs_CC", "Origen_CC", "Colgo_Agente_CC", "Salida_CC", "Campo_Clave", "acw",
        "Calificacion_Int_CC", "Clave_int_cli"
    ]

    if not api_key or not sys_code:
        return pd.DataFrame(columns=TARGET_COLS)

    pdate_text = pdate.strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            response = requests.get(
                API_URL,
                params={
                    "service": "cc/api",
                    "m": "27",
                    "key": api_key,
                    "sys": sys_code,
                    "fecha_ini": pdate_text,
                    "fecha_fin": pdate_text,
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()

            records = payload.get("result", [])
            if not isinstance(records, list) or len(records) == 0:
                pytime.sleep(1 + attempt)
                continue

            df = pd.DataFrame(records)

            for c in base_cols:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[base_cols].copy()

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

        except Exception:
            pytime.sleep(1 + attempt)

    return pd.DataFrame(columns=TARGET_COLS)


def dedupe_glos_like_powerbi(df: pd.DataFrame, overwrite_fecha: datetime | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    out["Estatus_CC"] = out["Estatus_CC"].fillna("").astype(str).str.strip()
    out["Estatus_CC"] = np.where(out["Estatus_CC"] == "", "SIN CALIFICACION", out["Estatus_CC"])
    out["Estatus_CC"] = out["Estatus_CC"].str.upper()

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

    out["Tipificacion_Detalle"] = out["Estatus_CC"]
    out["Tipificacion_3"] = out["Estatus_CC"].apply(status_group_3)
    out["Tipificacion_3_Abbr"] = out["Tipificacion_3"].map(TIP_ABBR).fillna("OTR")

    for c in ["Gestor_CC", "Calificacion_Int_CC", "Extension_CC"]:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str).str.strip()

    return out.reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_glos_yesterday(api_key: str, sys_code: str) -> tuple[pd.DataFrame, date]:
    hoy = mexico_now().date()

    dia_vencido = compute_business_reference_day(hoy)
    target = compute_business_reference_day(dia_vencido)

    raw0 = fetch_glos_raw(target, api_key, sys_code)
    if raw0.empty:
        alt = compute_business_reference_day(target)
        raw0 = fetch_glos_raw(alt, api_key, sys_code)
        target = alt

    return dedupe_glos_like_powerbi(raw0), target


@st.cache_data(ttl=60, show_spinner=False)
def get_glos_today_visual(api_key: str, sys_code: str) -> tuple[pd.DataFrame, date]:
    now_cdmx = mexico_now()
    hoy = now_cdmx.date()

    day_vencido = compute_business_reference_day(hoy)
    raw = fetch_glos_raw(day_vencido, api_key, sys_code)

    if raw.empty:
        alt = day_vencido - timedelta(days=1)
        raw = fetch_glos_raw(alt, api_key, sys_code)
        day_vencido = alt

    out = dedupe_glos_like_powerbi(raw, overwrite_fecha=now_cdmx)
    return out, day_vencido


@st.cache_data(ttl=60, show_spinner=False)
def get_glos_exact_day(pdate: date, api_key: str, sys_code: str) -> pd.DataFrame:
    raw = fetch_glos_raw(pdate, api_key, sys_code)
    return dedupe_glos_like_powerbi(raw)


# -------------------------------
# EXTRA KPI HELPERS
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
    work = df.copy()

    if team_col not in work.columns:
        work[team_col] = "Sin equipo"
    if agent_col not in work.columns:
        work[agent_col] = "Sin agente"

    work[team_col] = work[team_col].replace("", np.nan).fillna("Sin equipo")
    work[agent_col] = work[agent_col].replace("", np.nan).fillna("Sin agente")

    team = (
        work.groupby([team_col, tip_col], dropna=False)
        .size()
        .reset_index(name="Registros")
    )

    agent = (
        work.groupby([agent_col, tip_col], dropna=False)
        .size()
        .reset_index(name="Registros")
    )

    return team, agent


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
        st.error("Faltan credenciales GLOS en .streamlit/secrets.toml")

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
    st.title("Dashboard interactivo de tipificación - GLOS")
    st.warning("Agrega tus credenciales en `.streamlit/secrets.toml` para cargar la información.")
    st.stop()


# -------------------------------
# TITLE
# -------------------------------
st.title("Tipificación - GLOS")
st.caption("Desglose diario por tipificación con detalle por equipo y por agente.")


# -------------------------------
# LOAD DATA
# -------------------------------
if view_mode == "Hoy":
    df, source_date = get_glos_today_visual(GLOS_API_KEY, GLOS_SYS)
    visual_label = f"Hoy ({mexico_now().strftime('%Y-%m-%d %H:%M:%S')})"
    source_label = f"Datos base del día hábil: {source_date}"
elif view_mode == "Día hábil anterior":
    df, source_date = get_glos_yesterday(GLOS_API_KEY, GLOS_SYS)
    visual_label = f"Día hábil anterior ({source_date})"
    source_label = f"Datos reales de: {source_date}"
else:
    df = get_glos_exact_day(selected_date, GLOS_API_KEY, GLOS_SYS)
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

work_kpi = work.copy()

if selected_tip:
    work = work[work[tip_col].isin(selected_tip)].copy()
else:
    work = work.iloc[0:0].copy()

if work.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

TEAM_LABEL = "Equipo"
AGENT_LABEL = "Agente"


# -------------------------------
# KPIS
# -------------------------------
total_reg = len(work)

contactos_vis = int((work["Tipificacion_3"] == "CONTACTO").sum())
improcedentes_vis = int((work["Tipificacion_3"] == "IMPROCEDENTE").sum())
no_contactados_vis = int((work["Tipificacion_3"] == "NO CONTACTADO").sum())
otros_vis = int((work["Tipificacion_3"] == "OTROS / SIN CALIFICACION").sum())

avg_contacto = avg_active_seconds_by_tip(work_kpi, "CONTACTO")
avg_improcedente = avg_active_seconds_by_tip(work_kpi, "IMPROCEDENTE")
avg_no_contactado = avg_active_seconds_by_tip(work_kpi, "NO CONTACTADO")

agent_hangup_pct = calc_agent_hangup_pct(work_kpi)
discard_block_count = calc_discard_block_count(work)
awc_value = calc_awc(work_kpi)


c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Registros", f"{total_reg:,}")

with c2:
    st.metric("Contacto (avg sec)", f"{avg_contacto:,.1f}")


with c3:
    st.metric("Improcedente (avg sec)", f"{avg_improcedente:,.1f}")


with c4:
    st.metric("No contactado (avg sec)", f"{avg_no_contactado:,.1f}")


with c5:
    st.metric("Otros / sin calificación", f"{otros_vis:,}")


st.caption(f"Vista: {visual_label} | {source_label}")

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("% Colgó Agente", f"{agent_hangup_pct:.0f}%")


with k2:
    st.metric("Bloqueo de Discard", f"{discard_block_count:,}")


with k3:
    st.metric("AWC", f"{awc_value:,}")



# -------------------------------
# SUMMARIES
# -------------------------------
summary_tip = build_summary(work, tip_col)

if tip_col == "Tipificacion_3":
    summary_tip["OrdenTmp"] = summary_tip[tip_col].map({k: i for i, k in enumerate(TIP_ORDER_3)}).fillna(999)
    summary_tip = summary_tip.sort_values(["OrdenTmp", "Registros"], ascending=[True, False]).drop(columns="OrdenTmp")

team_summary_long, agent_summary_long = build_team_agent_summary(work, tip_col, team_col, agent_col)

team_pivot = (
    team_summary_long.pivot(index=team_col, columns=tip_col, values="Registros")
    .fillna(0)
    .reset_index()
)

agent_pivot = (
    agent_summary_long.pivot(index=agent_col, columns=tip_col, values="Registros")
    .fillna(0)
    .reset_index()
)


# -------------------------------
# CHARTS
# -------------------------------
left, right = st.columns([0.95, 1.05])

with left:
    fig_tip = px.bar(
        summary_tip,
        x=tip_col,
        y="Registros",
        text="Registros",
        title="Desglose por tipificación",
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
        title="Participación por tipificación",
    )
    fig_donut.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("### Desglose por equipo")

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
    title="Tipificación por equipo",
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

st.markdown("### Desglose por agente")

top_agents = (
    work.groupby(agent_col)
    .size()
    .sort_values(ascending=False)
    .head(25)
    .index.tolist()
)

agent_chart_df = agent_summary_long[
    agent_summary_long[agent_col].isin(top_agents)
].copy()

agent_chart_df = agent_chart_df.sort_values(
    [agent_col, "Registros"],
    ascending=[True, False]
)

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
    title="Top agentes por volumen",
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
    "Resumen general",
    "Por equipo",
    "Por agente",
    "Detalle de registros"
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
    agent_show = agent_pivot.copy().rename(columns={agent_col: AGENT_LABEL})
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
excel_agent = agent_pivot.copy().rename(columns={agent_col: AGENT_LABEL})

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