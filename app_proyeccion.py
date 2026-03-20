import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OXXO | Proyección Espejo",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --red: #ED1C24;
  --red-dark: #B01318;
  --gold: #FFD100;
  --dark: #0D0D0D;
  --dark2: #1A1A1A;
  --dark3: #252525;
  --surface: #1E1E1E;
  --border: rgba(255,255,255,0.07);
  --text: #F0F0F0;
  --muted: #888;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--dark) !important;
  color: var(--text) !important;
}

.stApp { background: var(--dark) !important; }

/* HERO */
.hero {
  background: linear-gradient(135deg, #0D0D0D 0%, #1A0000 40%, #2A0000 100%);
  border: 1px solid rgba(237,28,36,0.25);
  border-radius: 16px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(237,28,36,0.15) 0%, transparent 70%);
  border-radius: 50%;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 200px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(255,209,0,0.08) 0%, transparent 70%);
  border-radius: 50%;
}
.hero h1 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3rem;
  color: white;
  margin: 0 0 0.3rem 0;
  letter-spacing: 2px;
  line-height: 1;
}
.hero h1 span { color: var(--gold); }
.hero p { color: #aaa; font-size: 1rem; margin: 0; font-weight: 300; }
.hero .badge {
  display: inline-block;
  background: rgba(237,28,36,0.2);
  border: 1px solid rgba(237,28,36,0.4);
  color: #ff6b6b;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 1rem;
}

/* METRIC CARDS */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(237,28,36,0.3); }
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--red), transparent);
}
.kpi-card.gold::before { background: linear-gradient(90deg, var(--gold), transparent); }
.kpi-card.green::before { background: linear-gradient(90deg, #22c55e, transparent); }
.kpi-card .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 0.5rem; }
.kpi-card .value { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: white; line-height: 1; }
.kpi-card .value.red { color: var(--red); }
.kpi-card .value.gold { color: var(--gold); }
.kpi-card .value.green { color: #22c55e; }
.kpi-card .sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }

/* HIGHLIGHT BOXES */
.proj-highlight {
  background: linear-gradient(135deg, rgba(237,28,36,0.08) 0%, rgba(237,28,36,0.03) 100%);
  border: 1px solid rgba(237,28,36,0.25);
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.proj-highlight.gold-h {
  background: linear-gradient(135deg, rgba(255,209,0,0.08) 0%, rgba(255,209,0,0.03) 100%);
  border-color: rgba(255,209,0,0.25);
}
.proj-highlight .big-num {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.5rem;
  line-height: 1;
  margin: 0.5rem 0;
}
.proj-highlight .big-num.red-num { color: var(--red); }
.proj-highlight .big-num.gold-num { color: var(--gold); }
.proj-highlight .label-h { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
.proj-highlight .sub-h { font-size: 0.85rem; color: #888; margin-top: 0.3rem; }

/* MIRROR CARD */
.mirror-card {
  background: linear-gradient(135deg, var(--surface) 0%, rgba(37,37,37,0.8) 100%);
  border: 1px solid rgba(255,209,0,0.2);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
}
.mirror-icon {
  width: 56px; height: 56px;
  background: linear-gradient(135deg, rgba(255,209,0,0.15), rgba(255,209,0,0.05));
  border: 1px solid rgba(255,209,0,0.3);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem;
  flex-shrink: 0;
}
.mirror-info .title { font-weight: 700; font-size: 1.1rem; color: white; }
.mirror-info .sub { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }
.sim-badge {
  margin-left: auto;
  background: rgba(255,209,0,0.15);
  border: 1px solid rgba(255,209,0,0.3);
  color: var(--gold);
  padding: 0.5rem 1.2rem;
  border-radius: 30px;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.5rem;
  letter-spacing: 1px;
}

/* RULE PILL */
.rule-pill {
  display: inline-flex; align-items: center; gap: 0.5rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  padding: 0.4rem 0.9rem;
  font-size: 0.78rem;
  color: #aaa;
  margin: 0.2rem;
}
.rule-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--gold); flex-shrink: 0; }

/* SIDEBAR */
[data-testid="stSidebar"] {
  background: var(--dark2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--gold) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--red) !important; }

/* BUTTON */
.stButton > button {
  background: linear-gradient(135deg, var(--red) 0%, var(--red-dark) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.75rem 2rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.5px !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 4px 20px rgba(237,28,36,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(237,28,36,0.5) !important;
}

/* INPUTS */
.stSelectbox > div > div, .stNumberInput > div > div > input {
  background: var(--dark3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label { color: #aaa !important; font-size: 0.82rem !important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--dark2) !important;
  border-radius: 10px;
  padding: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: #888 !important;
  border-radius: 7px !important;
  padding: 8px 18px !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: var(--red) !important;
  color: white !important;
}

/* DATAFRAME */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

.stDownloadButton > button {
  background: rgba(255,209,0,0.12) !important;
  color: var(--gold) !important;
  border: 1px solid rgba(255,209,0,0.3) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: rgba(255,209,0,0.2) !important; }

/* DIVIDER */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* WARNING / SUCCESS */
.stAlert { border-radius: 10px !important; }

/* INFO BOX */
.info-box {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  font-size: 0.85rem;
  color: #ccc;
  margin: 1rem 0;
}
.section-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.4rem;
  letter-spacing: 2px;
  color: white;
  margin: 1.5rem 0 1rem 0;
  display: flex; align-items: center; gap: 0.6rem;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(237,28,36,0.3), transparent);
}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class='hero'>
  <div class='badge'>Motor de Proyección v2.0</div>
  <h1>🏪 OXXO <span>ESPEJO</span></h1>
  <p>Proyecta Ventas Operativas y Contribución Directa a 30 meses · Tiendas nuevas ≤ 10 meses</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
MESES_MAP = {
    'ene':'01','feb':'02','mar':'03','abr':'04','may':'05','jun':'06',
    'jul':'07','ago':'08','sep':'09','oct':'10','nov':'11','dic':'12'
}

def parse_fecha(s):
    try:
        parts = str(s).strip().lower().split()
        if len(parts) == 2:
            m = MESES_MAP.get(parts[0], '01')
            y = '20' + parts[1] if len(parts[1]) == 2 else parts[1]
            return pd.Timestamp(f"{y}-{m}-01")
    except: pass
    return pd.NaT


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    xls = pd.ExcelFile(file_bytes)

    # INFO TIENDAS
    df_info = xls.parse('INFO TIENDAS')
    df_info.columns = [str(c).strip().upper() for c in df_info.columns]
    df_info = df_info.rename(columns={c: 'RENTA' for c in df_info.columns if 'RENTA' in c})
    if 'VT' in df_info.columns: df_info['VIVIENDAS'] = df_info['VT']
    if 'ET' in df_info.columns: df_info['EMPLEOS'] = df_info['ET']
    for col in ['VIVIENDAS','EMPLEOS','AREA','ESTRATO']:
        if col not in df_info.columns: df_info[col] = 0
    if 'SEG26' not in df_info.columns: df_info['SEG26'] = 'BASE'
    df_info['CR'] = df_info['CR'].astype(str).str.strip()

    def parse_sheet(sheet_name, value_col='Ventas'):
        df_raw = xls.parse(sheet_name, header=None)
        header_row = df_raw.iloc[5, :]
        date_labels = header_row[1:].tolist()
        df_piv = df_raw.iloc[6:, :].copy()
        df_piv.columns = ['CR'] + date_labels
        df_piv = df_piv.reset_index(drop=True)
        df_piv['CR'] = df_piv['CR'].astype(str).str.strip()
        if 'Total general' in df_piv.columns:
            df_piv = df_piv.drop(columns=['Total general'])
        df_v = df_piv.melt(id_vars=['CR'], var_name='Fecha_Raw', value_name=value_col)
        df_v = df_v.dropna(subset=[value_col])
        df_v[value_col] = pd.to_numeric(df_v[value_col], errors='coerce')
        df_v = df_v.dropna(subset=[value_col])
        df_v['Fecha'] = df_v['Fecha_Raw'].apply(parse_fecha)
        df_v = df_v.dropna(subset=['Fecha'])
        df_v = df_v.sort_values(['CR','Fecha']).reset_index(drop=True)
        df_v['Mes_Num'] = df_v.groupby('CR').cumcount() + 1
        return df_v

    df_ventas = parse_sheet('HISTORICO VENTAS', 'Ventas')
    df_contrib = parse_sheet('HISTORICO CON', 'Contribucion')

    return df_ventas, df_contrib, df_info


# ─────────────────────────────────────────────────────────
# MIRROR STORE — últimos 3 meses promedio >= 230,000
# ─────────────────────────────────────────────────────────
def find_mirror(df_info, nueva, df_ventas, exclude_cr, min_months, pesos, min_ventas_ult3):
    df_f = df_info.copy()
    df_f = df_f[df_f['CR'] != exclude_cr]

    meses_cr = df_ventas.groupby('CR')['Mes_Num'].max().reset_index()
    meses_cr.columns = ['CR', 'total_meses']
    df_f = df_f.merge(meses_cr, on='CR', how='left')
    df_f['total_meses'] = df_f['total_meses'].fillna(0)
    df_f = df_f[df_f['total_meses'] >= min_months]

    if df_f.empty:
        return None, f"Sin tiendas con ≥{min_months} meses"

    # Promedio últimos 3 meses
    def prom_ult3(cr):
        s = df_ventas[df_ventas['CR'] == cr].sort_values('Mes_Num')['Ventas']
        if len(s) < 1: return 0
        return s.tail(3).mean()

    df_f = df_f.copy()
    df_f['ventas_ult3'] = df_f['CR'].map(prom_ult3).fillna(0)
    n_antes = len(df_f)
    df_f = df_f[df_f['ventas_ult3'] >= min_ventas_ult3]

    if df_f.empty:
        return None, (
            f"Sin tiendas espejo con promedio últimos 3 meses ≥ ${min_ventas_ult3:,.0f} "
            f"y ≥{min_months} meses. ({n_antes} candidatas antes del filtro)")

    # Distancia euclidiana ponderada
    vars_num = ['ESTRATO','AREA','VIVIENDAS','EMPLEOS']
    X_num = df_f[vars_num].fillna(0).values
    X_new = np.array([[nueva.get(v, 0) for v in vars_num]])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_num)
    Xn_sc = scaler.transform(X_new)

    vars_cat = ['ZONA','TIPO DE LOCAL','GENERADOR','MUN']
    X_cat = np.zeros((len(df_f), len(vars_cat)))
    for i, var in enumerate(vars_cat):
        if var in df_f.columns:
            X_cat[:, i] = (df_f[var].astype(str) == str(nueva.get(var, ''))).astype(int)
    Xn_cat = np.ones((1, len(vars_cat)))

    X_full = np.hstack([X_sc, X_cat])
    Xn_full = np.hstack([Xn_sc, Xn_cat])
    pw = np.sqrt(np.array([pesos.get(k, 0) for k in
                            ['ESTRATO','AREA','VIVIENDAS','EMPLEOS',
                             'ZONA','TIPO DE LOCAL','GENERADOR','MUN']]))

    dist = euclidean_distances(Xn_full * pw, X_full * pw)[0]
    mn, mx = dist.min(), dist.max()
    sim = (1 - (dist - mn) / (mx - mn + 1e-9)) * 100

    df_f = df_f.copy()
    df_f['DISTANCIA'] = dist
    df_f['SIMILITUD'] = sim

    # Ventas prom general también
    ventas_prom = df_ventas.groupby('CR')['Ventas'].mean().reset_index()
    ventas_prom.columns = ['CR', 'ventas_prom']
    df_f = df_f.merge(ventas_prom, on='CR', how='left')

    return df_f.sort_values('SIMILITUD', ascending=False), None


# ─────────────────────────────────────────────────────────
# PROJECTION — VENTAS (multiplicativo, clip>=0)
# ─────────────────────────────────────────────────────────
def project_series(new_sales_raw, mirror_sales_all, target=30, model_name='poly2'):
    mirror_sales = mirror_sales_all[:30]
    new_sales = new_sales_raw[1:] if len(new_sales_raw) > 1 else new_sales_raw

    X_m = np.arange(1, len(mirror_sales)+1).reshape(-1, 1)
    y_m = np.array(mirror_sales, dtype=float)

    models = {
        'linear': LinearRegression().fit(X_m, y_m),
        'poly2': Pipeline([('p', PolynomialFeatures(2)), ('r', LinearRegression())]).fit(X_m, y_m),
        'poly3': Pipeline([('p', PolynomialFeatures(3)), ('r', LinearRegression())]).fit(X_m, y_m),
    }
    r2s = {n: r2_score(y_m, m.predict(X_m)) for n, m in models.items()}

    n_ov = min(len(new_sales), len(mirror_sales))
    scale = (np.mean(new_sales[:n_ov]) / (np.mean(mirror_sales[:n_ov]) + 1e-9)) if n_ov >= 2 \
        else (new_sales[0] / (mirror_sales[0] + 1e-9) if new_sales else 1.0)

    X_all = np.arange(1, target+1).reshape(-1, 1)
    proj = np.clip(models[model_name].predict(X_all) * scale, 0, None)
    cur = len(new_sales)

    df_res = pd.DataFrame({
        'Mes': range(1, target+1),
        'Valor': [new_sales[i] if i < cur else proj[i] for i in range(target)],
        'Tipo': ['Real' if i < cur else 'Proyectado' for i in range(target)],
    })
    metrics = {
        'scale': scale, 'r2': r2s[model_name], 'r2s': r2s,
        'm28': proj[27] if target >= 28 else None,
        'm29': proj[28] if target >= 29 else None,
        'm30': proj[29] if target >= 30 else None,
        'prom_28_30': np.mean([proj[i] for i in [27, 28, 29] if i < target]),
        'meses_reales_usados': cur,
        'meses_espejo_usados': len(mirror_sales),
    }
    return df_res, metrics


# ─────────────────────────────────────────────────────────
# PROJECTION — CONTRIBUCION (aditivo con suavizado)
#
# La contribución tiene valores negativos por naturaleza
# (meses de inicio, estacionalidad, costos fijos).
# Estrategia:
#  1. Suavizar el espejo con media móvil (ventana 3) para
#     reducir el ruido y dar una curva de maduración limpia.
#  2. Ajustar con offset ADITIVO (no multiplicativo): calculado
#     como la diferencia promedio entre la nueva tienda y el
#     espejo en los meses coincidentes. Esto evita que signos
#     opuestos generen un factor de escala inválido.
#  3. El offset se aplana gradualmente de 100% → 30% en
#     24 meses, capturando la diferencia estructural de costos
#     sin que la tienda converja exactamente al espejo.
#  4. SIN clip: la contribución puede ser negativa y eso es
#     información válida (tienda aún en curva de inversión).
# ─────────────────────────────────────────────────────────
def project_contrib(new_contrib_raw, mirror_contrib_all, target=30, model_name='poly2'):
    mirror = np.array(mirror_contrib_all[:30], dtype=float)
    new_data = np.array(new_contrib_raw[1:] if len(new_contrib_raw) > 1 else new_contrib_raw,
                        dtype=float)

    # Suavizar espejo (media móvil centrada, ventana 3)
    mirror_smooth = (pd.Series(mirror)
                     .rolling(3, min_periods=1, center=True)
                     .mean().values)

    X_m = np.arange(1, len(mirror_smooth)+1).reshape(-1, 1)
    y_m = mirror_smooth

    models = {
        'linear': LinearRegression().fit(X_m, y_m),
        'poly2': Pipeline([('p', PolynomialFeatures(2)), ('r', LinearRegression())]).fit(X_m, y_m),
        'poly3': Pipeline([('p', PolynomialFeatures(3)), ('r', LinearRegression())]).fit(X_m, y_m),
    }
    r2s = {n: r2_score(y_m, m.predict(X_m)) for n, m in models.items()}

    X_all = np.arange(1, target+1).reshape(-1, 1)
    mirror_proj = models[model_name].predict(X_all)  # sin clip

    # Offset aditivo: diferencia promedio nueva − espejo en overlap
    n_ov = min(len(new_data), len(mirror_smooth))
    if n_ov >= 2:
        offset = float(np.mean(new_data[:n_ov] - mirror_proj[:n_ov]))
    elif len(new_data) == 1:
        offset = float(new_data[0] - mirror_proj[0])
    else:
        offset = 0.0

    # Taper: 1.0 → 0.3 en 24 meses (mantiene diferencia estructural residual)
    taper = np.array([max(0.3, 1.0 - i * 0.7 / 24) for i in range(target)])
    proj = mirror_proj + offset * taper

    cur = len(new_data)
    df_res = pd.DataFrame({
        'Mes': range(1, target+1),
        'Valor': [float(new_data[i]) if i < cur else float(proj[i]) for i in range(target)],
        'Tipo': ['Real' if i < cur else 'Proyectado' for i in range(target)],
    })
    metrics = {
        'offset': offset, 'r2': r2s[model_name], 'r2s': r2s,
        'm28': float(proj[27]) if target >= 28 else None,
        'm29': float(proj[28]) if target >= 29 else None,
        'm30': float(proj[29]) if target >= 30 else None,
        'prom_28_30': float(np.mean([proj[i] for i in [27, 28, 29] if i < target])),
        'meses_reales_usados': cur,
        'meses_espejo_usados': len(mirror),
        'mirror_smooth': mirror_smooth.tolist(),
    }
    return df_res, metrics


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Datos")

    uploaded = st.file_uploader("LibroVentas.xlsx", type=['xlsx', 'xls'])
    usar_local = False
    if uploaded is None and os.path.exists('LibroVentas.xlsx'):
        usar_local = st.checkbox("Usar archivo del repositorio", value=True)

    st.markdown("---")
    st.markdown("## ⚙️ Parámetros")

    min_months = st.slider("Meses mínimos espejo", 12, 36, 18,
                            help="Solo tiendas con ≥ N meses son candidatas espejo")
    target_months = st.slider("Meses a proyectar", 24, 36, 30)
    model_choice = st.selectbox("Modelo de proyección",
                                 ['poly2', 'poly3', 'linear'],
                                 format_func=lambda x: {
                                     'poly2': 'Polinomial grado 2 ✅',
                                     'poly3': 'Polinomial grado 3',
                                     'linear': 'Lineal'}[x])

    st.markdown("---")
    st.markdown("## 💰 Filtro Espejo")
    min_ventas_ult3 = st.number_input(
        "Prom. últimos 3 meses mínimo ($)",
        min_value=0, value=230_000, step=10_000,
        help="Solo se usan como espejo tiendas cuyo promedio de ventas de los últimos 3 meses sea ≥ este valor"
    )
    st.markdown(f"""
    <div style='background:rgba(255,209,0,0.08);border:1px solid rgba(255,209,0,0.2);
    border-radius:8px;padding:0.7rem 1rem;font-size:0.78rem;color:#ccc;margin-top:0.5rem;'>
    🎯 Espejo: ≥{min_months} meses<br>
    📊 Últ. 3 meses prom ≥ <b style='color:#FFD100'>${min_ventas_ult3:,.0f}</b><br>
    ⚡ Nueva: se descarta mes 1
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚖️ Pesos Similitud")
    p = {
        'ZONA':          st.slider("Zona",           0, 100, 15),
        'ESTRATO':       st.slider("Estrato",         0, 100, 12),
        'AREA':          st.slider("Área m²",         0, 100, 12),
        'TIPO DE LOCAL': st.slider("Tipo de Local",   0, 100, 12),
        'GENERADOR':     st.slider("Generador",       0, 100, 10),
        'VIVIENDAS':     st.slider("Viviendas (VT)",  0, 100, 8),
        'EMPLEOS':       st.slider("Empleos (ET)",    0, 100, 8),
        'MUN':           st.slider("Municipio",       0, 100, 8),
    }
    total_p = sum(p.values()) or 1
    pesos = {k: v / total_p * 0.70 for k, v in p.items()}
    pesos['SEG26'] = 0.30


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
data_bytes = None
if uploaded:
    data_bytes = uploaded.read()
elif usar_local:
    with open('LibroVentas.xlsx', 'rb') as f:
        data_bytes = f.read()

if data_bytes is None:
    st.markdown("""
    <div class='info-box'>
    <b>👈 Carga tu archivo para comenzar</b><br><br>
    El archivo debe contener <b>3 hojas</b>:<br>
    • <code>INFO TIENDAS</code> → CR, NAME, ZONA, MUN, ESTRATO, TIPO DE LOCAL, AREA, SEG26, GENERADOR, VT, ET<br>
    • <code>HISTORICO VENTAS</code> → Pivote CR × mes de Ventas Operativas<br>
    • <code>HISTORICO CON</code> → Pivote CR × mes de Contribución Directa
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Cargando y procesando datos..."):
    df_ventas, df_contrib, df_info = load_data(data_bytes)

# Stats
meses_cr = df_ventas.groupby('CR')['Mes_Num'].max().reset_index()
meses_cr.columns = ['CR', 'total_meses']

def prom_ult3_fn(cr):
    s = df_ventas[df_ventas['CR'] == cr].sort_values('Mes_Num')['Ventas']
    return s.tail(3).mean() if len(s) >= 1 else 0

tiendas_nuevas = meses_cr[meses_cr['total_meses'] <= 10].copy()
tiendas_nuevas = tiendas_nuevas.merge(
    df_info[['CR', 'NAME']].drop_duplicates(), on='CR', how='left')

candidatos_espejo = meses_cr[meses_cr['total_meses'] >= min_months].copy()
candidatos_espejo['ult3'] = candidatos_espejo['CR'].map(prom_ult3_fn)
candidatos_espejo = candidatos_espejo[candidatos_espejo['ult3'] >= min_ventas_ult3]

# KPI Summary
st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-card'>
    <div class='label'>Tiendas en Base</div>
    <div class='value red'>{df_ventas['CR'].nunique()}</div>
    <div class='sub'>histórico total</div>
  </div>
  <div class='kpi-card'>
    <div class='label'>Info Cargada</div>
    <div class='value'>{df_info['CR'].nunique()}</div>
    <div class='sub'>tiendas con características</div>
  </div>
  <div class='kpi-card gold'>
    <div class='label'>Nuevas ≤ 10 meses</div>
    <div class='value gold'>{len(tiendas_nuevas)}</div>
    <div class='sub'>para proyectar</div>
  </div>
  <div class='kpi-card green'>
    <div class='label'>Candidatos Espejo</div>
    <div class='value green'>{len(candidatos_espejo)}</div>
    <div class='sub'>≥{min_months}m · últ3 ≥${min_ventas_ult3/1000:.0f}K</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.4], gap="large")

with col_left:
    st.markdown("<div class='section-title'>🏪 Tienda Nueva</div>", unsafe_allow_html=True)

    if tiendas_nuevas.empty:
        st.warning("No hay tiendas con ≤ 10 meses en el histórico.")
        st.stop()

    tiendas_nuevas['label'] = tiendas_nuevas.apply(
        lambda r: f"{r['CR']}  —  {str(r.get('NAME','?'))[:20]}  ({int(r['total_meses'])}m)", axis=1)

    sel_label = st.selectbox("Seleccionar tienda:", tiendas_nuevas['label'].tolist(), label_visibility='collapsed')
    sel_row = tiendas_nuevas[tiendas_nuevas['label'] == sel_label].iloc[0]
    sel_cr = sel_row['CR']
    sel_meses_total = int(sel_row['total_meses'])
    sel_meses_usados = max(0, sel_meses_total - 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Meses registrados", sel_meses_total)
    c2.metric("Meses útiles", sel_meses_usados, help="Sin mes 1")
    c3.metric("Meses a proyectar", target_months - sel_meses_usados)

    row_info = df_info[df_info['CR'] == sel_cr]
    has_info = len(row_info) > 0

    def get_val(col, default=0):
        if has_info and col in row_info.columns:
            v = row_info.iloc[0][col]
            return default if pd.isna(v) else v
        return default

    def safe_idx(lst, val):
        try: return list(lst).index(val)
        except: return 0

    all_segs  = sorted(df_info['SEG26'].dropna().unique())
    all_zonas = sorted(df_info['ZONA'].dropna().unique())
    all_muns  = sorted(df_info['MUN'].dropna().unique())
    all_tipos = sorted(df_info['TIPO DE LOCAL'].dropna().unique())
    all_gens  = sorted(df_info['GENERADOR'].dropna().unique())
    all_estr  = sorted(df_info['ESTRATO'].dropna().unique())

    st.markdown("<div class='section-title' style='font-size:1rem;'>Características</div>", unsafe_allow_html=True)
    st.caption("Pre-cargadas desde INFO TIENDAS")

    seg26    = st.selectbox("Segmento (SEG26)", all_segs,   index=safe_idx(all_segs,   get_val('SEG26',   'BASE')))
    zona     = st.selectbox("Zona",             all_zonas,  index=safe_idx(all_zonas,  get_val('ZONA',    '')))
    mun      = st.selectbox("Municipio",        all_muns,   index=safe_idx(all_muns,   get_val('MUN',     '')))
    estrato  = st.selectbox("Estrato",          all_estr,   index=safe_idx(all_estr,   get_val('ESTRATO', 3)))
    tipo_loc = st.selectbox("Tipo de Local",    all_tipos,  index=safe_idx(all_tipos,  get_val('TIPO DE LOCAL', '')))
    generador= st.selectbox("Generador",        all_gens,   index=safe_idx(all_gens,   get_val('GENERADOR', '')))

    ca, cb = st.columns(2)
    with ca:
        area      = st.number_input("Área (m²)",      value=float(get_val('AREA', 100)), step=5.0)
        viviendas = st.number_input("Viviendas (VT)", value=int(get_val('VT', 1000)),    step=100)
    with cb:
        empleos   = st.number_input("Empleos (ET)",   value=int(get_val('ET', 500)),     step=50)

    nueva_dict = {
        'CR': sel_cr, 'SEG26': seg26, 'ZONA': zona, 'MUN': mun,
        'ESTRATO': estrato, 'TIPO DE LOCAL': tipo_loc, 'GENERADOR': generador,
        'AREA': area, 'VIVIENDAS': viviendas, 'EMPLEOS': empleos,
    }

    run = st.button("🚀  Proyectar Ventas + Contribución", use_container_width=True)


# ─────────────────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────────────────
with col_right:
    st.markdown("<div class='section-title'>📊 Resultados</div>", unsafe_allow_html=True)

    if not run:
        st.markdown(f"""
        <div class='info-box'>
        <b>ℹ️ Cómo funciona:</b><br>
        1 · Selecciona tienda nueva (≤10 meses) en el panel izquierdo<br>
        2 · Verifica características — auto-cargadas desde INFO TIENDAS<br>
        3 · Pulsa <b>🚀 Proyectar</b> para obtener Ventas Operativas y Contribución Directa
        </div>
        <div style='margin-top:0.8rem;display:flex;flex-wrap:wrap;gap:0.4rem;'>
          <span class='rule-pill'><span class='dot'></span>Espejo ≥{min_months} meses</span>
          <span class='rule-pill'><span class='dot'></span>Últ. 3 meses prom ≥ ${min_ventas_ult3:,.0f}</span>
          <span class='rule-pill'><span class='dot'></span>Se usan primeros 30m del espejo</span>
          <span class='rule-pill'><span class='dot'></span>Mes 1 de la tienda nueva descartado</span>
          <span class='rule-pill'><span class='dot'></span>Proyección a {target_months} meses</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Tiendas disponibles para proyectar")
        disp = tiendas_nuevas[['CR', 'NAME', 'total_meses']].rename(
            columns={'total_meses': 'Meses', 'NAME': 'Nombre'})
        st.dataframe(disp, use_container_width=True, hide_index=True, height=340)
        st.stop()

    # ── Buscar espejo ──────────────────────────────────────
    with st.spinner("🔍 Buscando tienda espejo óptima..."):
        res_espejo, err = find_mirror(
            df_info, nueva_dict, df_ventas,
            exclude_cr=sel_cr, min_months=min_months,
            pesos=pesos, min_ventas_ult3=min_ventas_ult3)

    if err:
        st.error(f"❌ {err}")
        st.stop()

    mejor       = res_espejo.iloc[0]
    espejo_cr   = mejor['CR']
    espejo_name = str(mejor.get('NAME', espejo_cr))
    espejo_meses = int(mejor.get('total_meses', 0))
    espejo_ult3 = mejor.get('ventas_ult3', 0)
    espejo_vprom = mejor.get('ventas_prom', 0)

    # Mirror card
    st.markdown(f"""
    <div class='mirror-card'>
      <div class='mirror-icon'>🏪</div>
      <div class='mirror-info'>
        <div class='title'>{espejo_name} · {espejo_cr}</div>
        <div class='sub'>{espejo_meses} meses de historia · Prom. últ. 3m: <b style='color:#FFD100'>${espejo_ult3:,.0f}</b> · Prom. gral: ${espejo_vprom:,.0f}</div>
      </div>
      <div class='sim-badge'>⚡ {mejor['SIMILITUD']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Proyección VENTAS ──────────────────────────────────
    new_v_raw    = df_ventas[df_ventas['CR'] == sel_cr].sort_values('Mes_Num')['Ventas'].tolist()
    mirror_v_all = df_ventas[df_ventas['CR'] == espejo_cr].sort_values('Mes_Num')['Ventas'].tolist()
    proj_v_df, met_v = project_series(new_v_raw, mirror_v_all, target_months, model_choice)

    # ── Proyección CONTRIBUCION ────────────────────────────
    new_c_raw    = df_contrib[df_contrib['CR'] == sel_cr].sort_values('Mes_Num')['Contribucion'].tolist()
    mirror_c_all = df_contrib[df_contrib['CR'] == espejo_cr].sort_values('Mes_Num')['Contribucion'].tolist()
    has_contrib = len(new_c_raw) > 0 and len(mirror_c_all) > 0

    if has_contrib:
        proj_c_df, met_c = project_contrib(new_c_raw, mirror_c_all, target_months, model_choice)
    else:
        proj_c_df, met_c = None, None
        st.warning("⚠️ Sin datos de contribución para esta tienda o su espejo.")

    # ── KPIs 28-30 VENTAS & CONTRIBUCION ──────────────────
    st.markdown("---")
    st.markdown("""<div class='section-title' style='font-size:1.2rem;'>🎯 Proyección Meses 28 · 29 · 30</div>""",
                unsafe_allow_html=True)

    # Ventas row
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.markdown(f"""
        <div class='proj-highlight'>
          <div class='label-h'>💰 Ventas Mes 28</div>
          <div class='big-num red-num'>${met_v['m28']:,.0f}</div>
          <div class='sub-h'>Ventas Operativas</div>
        </div>""", unsafe_allow_html=True)
    with col_v2:
        st.markdown(f"""
        <div class='proj-highlight'>
          <div class='label-h'>💰 Ventas Mes 29</div>
          <div class='big-num red-num'>${met_v['m29']:,.0f}</div>
          <div class='sub-h'>Ventas Operativas</div>
        </div>""", unsafe_allow_html=True)
    with col_v3:
        st.markdown(f"""
        <div class='proj-highlight'>
          <div class='label-h'>💰 Ventas Mes 30</div>
          <div class='big-num red-num'>${met_v['m30']:,.0f}</div>
          <div class='sub-h'>Ventas Operativas</div>
        </div>""", unsafe_allow_html=True)
    with col_v4:
        st.markdown(f"""
        <div class='proj-highlight'>
          <div class='label-h'>📊 Prom 28–30 Ventas</div>
          <div class='big-num red-num'>${met_v['prom_28_30']:,.0f}</div>
          <div class='sub-h'>promedio meses 28, 29 y 30</div>
        </div>""", unsafe_allow_html=True)

    if has_contrib and met_c:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.markdown(f"""
            <div class='proj-highlight gold-h'>
              <div class='label-h'>🟡 Contribución Mes 28</div>
              <div class='big-num gold-num'>${met_c['m28']:,.0f}</div>
              <div class='sub-h'>Contribución Directa</div>
            </div>""", unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class='proj-highlight gold-h'>
              <div class='label-h'>🟡 Contribución Mes 29</div>
              <div class='big-num gold-num'>${met_c['m29']:,.0f}</div>
              <div class='sub-h'>Contribución Directa</div>
            </div>""", unsafe_allow_html=True)
        with col_c3:
            st.markdown(f"""
            <div class='proj-highlight gold-h'>
              <div class='label-h'>🟡 Contribución Mes 30</div>
              <div class='big-num gold-num'>${met_c['m30']:,.0f}</div>
              <div class='sub-h'>Contribución Directa</div>
            </div>""", unsafe_allow_html=True)
        with col_c4:
            st.markdown(f"""
            <div class='proj-highlight gold-h'>
              <div class='label-h'>📊 Prom 28–30 Contrib.</div>
              <div class='big-num gold-num'>${met_c['prom_28_30']:,.0f}</div>
              <div class='sub-h'>promedio meses 28, 29 y 30</div>
            </div>""", unsafe_allow_html=True)

        # Margen implícito
        margin = met_c['prom_28_30'] / (met_v['prom_28_30'] + 1e-9) * 100
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-radius:10px;padding:0.8rem 1.2rem;margin-top:0.8rem;
        display:flex;align-items:center;gap:1rem;font-size:0.85rem;color:#bbb;'>
          <span style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;color:#22c55e;'>{margin:.1f}%</span>
          <span>Margen implícito promedio (Contribución / Ventas) en meses 28–30 · 
          Factor escala ventas: <b style='color:#ED1C24'>{met_v['scale']:.3f}×</b> · 
          Factor escala contrib.: <b style='color:#FFD100'>{met_c['scale']:.3f}×</b> · 
          R² ventas: <b>{met_v['r2']:.4f}</b> · R² contrib.: <b>{met_c['r2']:.4f}</b></span>
        </div>
        """, unsafe_allow_html=True)

    # ── GRAFICOS ────────────────────────────────────────
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Ventas Operativas",
        "💛 Contribución Directa",
        "🏪 Candidatos Espejo",
        "📐 Comparar Modelos"
    ])

    PLOTLY_LAYOUT = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cccccc', family='DM Sans'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        margin=dict(t=60, b=40, l=40, r=40),
    )

    def make_projection_fig(proj_df, mirror_all, sel_cr, espejo_name, espejo_cr,
                             metrics, col_real, col_proj, col_mirror, title, yaxis_title):
        real_d = proj_df[proj_df['Tipo'] == 'Real']
        proy_d = proj_df[proj_df['Tipo'] == 'Proyectado']
        mirror_plot = mirror_all[:30]
        mir_df = pd.DataFrame({'Mes': range(1, len(mirror_plot)+1), 'Valor': mirror_plot})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mir_df['Mes'], y=mir_df['Valor'],
            name=f'Espejo: {espejo_name}',
            line=dict(color=col_mirror, width=1.5, dash='dot'),
            mode='lines', opacity=0.7))
        fig.add_trace(go.Scatter(
            x=real_d['Mes'], y=real_d['Valor'],
            name=f'{sel_cr} — Real',
            line=dict(color=col_real, width=3),
            mode='lines+markers', marker=dict(size=7, symbol='circle')))
        fig.add_trace(go.Scatter(
            x=proy_d['Mes'], y=proy_d['Valor'],
            name='Proyección',
            line=dict(color=col_proj, width=2.5, dash='dash'),
            fill='tozeroy', fillcolor=f'rgba({",".join(str(int(col_proj.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.05)'))

        for mes, val in [(28, metrics['m28']), (29, metrics['m29']), (30, metrics['m30'])]:
            if val and target_months >= mes:
                fig.add_vline(x=mes, line_color='rgba(255,209,0,0.4)', line_dash='dot')
                fig.add_annotation(x=mes, y=val,
                    text=f"<b>M{mes}</b><br>${val/1000:.0f}K",
                    showarrow=True, arrowhead=2, arrowcolor='#FFD100',
                    bgcolor='rgba(20,20,20,0.9)', bordercolor='#FFD100',
                    font=dict(color='#FFD100', size=11, family='DM Sans'),
                    borderpad=5, borderwidth=1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color='white')),
            xaxis_title='Mes de Operación',
            yaxis_title=yaxis_title,
            height=400,
            **PLOTLY_LAYOUT)
        return fig

    with tab1:
        fig_v = make_projection_fig(
            proj_v_df, mirror_v_all, sel_cr, espejo_name, espejo_cr, met_v,
            '#ED1C24', '#ff6b6b', '#555',
            f'Ventas Operativas — {sel_cr} · Modelo {model_choice}', 'Ventas ($)')
        st.plotly_chart(fig_v, use_container_width=True)

        col_tv1, col_tv2 = st.columns([2, 1])
        with col_tv1:
            disp_v = proj_v_df.copy()
            disp_v['Ventas ($)'] = disp_v['Valor'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(disp_v[['Mes', 'Ventas ($)', 'Tipo']],
                         use_container_width=True, height=250, hide_index=True)
        with col_tv2:
            csv_v = proj_v_df.to_csv(index=False)
            st.download_button("📥 CSV Ventas", csv_v,
                               f"ventas_{sel_cr}.csv", "text/csv",
                               use_container_width=True)

    with tab2:
        if has_contrib and proj_c_df is not None:
            # Build custom contribution chart with smoothed mirror and zero-line
            real_c  = proj_c_df[proj_c_df['Tipo'] == 'Real']
            proy_c  = proj_c_df[proj_c_df['Tipo'] == 'Proyectado']
            mirror_smooth_vals = met_c.get('mirror_smooth', mirror_c_all[:30])
            mir_c_df = pd.DataFrame({'Mes': range(1, len(mirror_smooth_vals)+1),
                                     'Valor': mirror_smooth_vals})

            fig_c = go.Figure()
            # Zero reference line
            fig_c.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_dash='dot',
                            annotation_text='Break-even', annotation_font_color='#888',
                            annotation_position='bottom right')
            # Espejo suavizado
            fig_c.add_trace(go.Scatter(
                x=mir_c_df['Mes'], y=mir_c_df['Valor'],
                name=f'Espejo suavizado: {espejo_name}',
                line=dict(color='#444', width=2, dash='dot'),
                mode='lines', opacity=0.8))
            # Real
            if len(real_c) > 0:
                fig_c.add_trace(go.Scatter(
                    x=real_c['Mes'], y=real_c['Valor'],
                    name=f'{sel_cr} — Real',
                    line=dict(color='#FFD100', width=3),
                    mode='lines+markers', marker=dict(size=7)))
            # Proyectado con fill positivo/negativo
            fig_c.add_trace(go.Scatter(
                x=proy_c['Mes'], y=proy_c['Valor'],
                name='Proyección contrib.',
                line=dict(color='#ffe566', width=2.5, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(255,209,0,0.07)'))
            # Anotaciones M28-30
            for mes, val in [(28, met_c['m28']), (29, met_c['m29']), (30, met_c['m30'])]:
                if val is not None and target_months >= mes:
                    fig_c.add_vline(x=mes, line_color='rgba(255,209,0,0.3)', line_dash='dot')
                    fig_c.add_annotation(x=mes, y=val,
                        text=f"<b>M{mes}</b><br>${val/1000:.0f}K",
                        showarrow=True, arrowhead=2, arrowcolor='#FFD100',
                        bgcolor='rgba(20,20,20,0.9)', bordercolor='#FFD100',
                        font=dict(color='#FFD100', size=11, family='DM Sans'),
                        borderpad=5, borderwidth=1)
            fig_c.update_layout(
                title=f'Contribución Directa — {sel_cr} · Espejo suavizado + offset aditivo · Modelo {model_choice}',
                xaxis_title='Mes de Operación',
                yaxis_title='Contribución ($)',
                height=420,
                **PLOTLY_LAYOUT)
            st.plotly_chart(fig_c, use_container_width=True)

            # Metodología note
            st.markdown(f"""
            <div class='info-box' style='font-size:0.8rem;'>
            <b>⚙️ Metodología contribución:</b> El espejo se suaviza con media móvil (ventana 3) para
            eliminar ruido. Se aplica un <b>offset aditivo de ${met_c['offset']:,.0f}</b> que refleja
            la diferencia estructural entre la nueva tienda y el espejo, tapering de 100% → 30% en 24 meses.
            Sin clip: valores negativos son válidos (tienda en curva de maduración).
            R² sobre espejo suavizado: <b>{met_c['r2']:.4f}</b>
            </div>
            """, unsafe_allow_html=True)

            col_tc1, col_tc2 = st.columns([2, 1])
            with col_tc1:
                disp_c = proj_c_df.copy()
                disp_c['Contribución ($)'] = disp_c['Valor'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(disp_c[['Mes', 'Contribución ($)', 'Tipo']],
                             use_container_width=True, height=250, hide_index=True)
            with col_tc2:
                csv_c = proj_c_df.to_csv(index=False)
                st.download_button("📥 CSV Contribución", csv_c,
                                   f"contrib_{sel_cr}.csv", "text/csv",
                                   use_container_width=True)

            # Combined chart — ventas + contribucion
            st.markdown("#### Ventas vs Contribución Directa (overlay)")
            fig_comb = go.Figure()
            fig_comb.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_dash='dot')
            fig_comb.add_trace(go.Scatter(
                x=proj_v_df['Mes'], y=proj_v_df['Valor'],
                name='Ventas', line=dict(color='#ED1C24', width=2.5),
                mode='lines',
                customdata=proj_v_df['Tipo'],
                hovertemplate='M%{x} | Ventas: $%{y:,.0f} (%{customdata})<extra></extra>'))
            fig_comb.add_trace(go.Scatter(
                x=proj_c_df['Mes'], y=proj_c_df['Valor'],
                name='Contribución', line=dict(color='#FFD100', width=2.5),
                mode='lines',
                customdata=proj_c_df['Tipo'],
                hovertemplate='M%{x} | Contrib: $%{y:,.0f} (%{customdata})<extra></extra>'))
            # Margin %
            pv = proj_v_df['Valor'].values
            pc = proj_c_df['Valor'].values
            meses_arr = proj_v_df['Mes'].values
            margin_pct = np.where(pv > 0, pc / pv * 100, 0)
            fig_comb.add_trace(go.Scatter(
                x=meses_arr, y=margin_pct,
                name='Margen %', yaxis='y2',
                line=dict(color='#22c55e', width=1.5, dash='dot'),
                hovertemplate='M%{x} | Margen: %{y:.1f}%<extra></extra>'))
            fig_comb.update_layout(
                yaxis2=dict(title='Margen (%)', overlaying='y', side='right',
                            gridcolor='rgba(255,255,255,0.03)',
                            tickfont=dict(color='#22c55e')),
                height=380,
                title='Ventas Operativas y Contribución Directa — 30 meses',
                xaxis_title='Mes', yaxis_title='$ Valor',
                **PLOTLY_LAYOUT)
            st.plotly_chart(fig_comb, use_container_width=True)
        else:
            st.info("Sin datos de contribución disponibles para esta tienda/espejo.")

    with tab3:
        cols_show = [c for c in ['CR', 'NAME', 'ZONA', 'MUN', 'ESTRATO', 'TIPO DE LOCAL',
                                   'AREA', 'SEG26', 'ventas_ult3', 'ventas_prom', 'SIMILITUD', 'total_meses']
                     if c in res_espejo.columns]
        top10 = res_espejo.head(10)[cols_show].copy()
        if 'SIMILITUD'   in top10.columns: top10['SIMILITUD']   = top10['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
        if 'ventas_ult3' in top10.columns: top10['ventas_ult3'] = top10['ventas_ult3'].apply(lambda x: f"${x:,.0f}")
        if 'ventas_prom' in top10.columns: top10['ventas_prom'] = top10['ventas_prom'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top10.rename(columns={
            'total_meses': 'Meses', 'ventas_ult3': 'Prom Últ.3', 'ventas_prom': 'Prom Gral'}),
            use_container_width=True, hide_index=True)

        fig_sim = px.bar(
            res_espejo.head(10), x='CR', y='SIMILITUD',
            title=f'Top 10 candidatos espejo · Prom. últ.3m ≥ ${min_ventas_ult3:,.0f}',
            color='SIMILITUD',
            color_continuous_scale=['#1a1a1a', '#B01318', '#ED1C24', '#FFD100'],
            hover_data=['ventas_ult3', 'total_meses'])
        fig_sim.update_layout(height=320, **PLOTLY_LAYOUT,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab4:
        summary = []
        for mn in ['linear', 'poly2', 'poly3']:
            _, met_m = project_series(new_v_raw, mirror_v_all, target_months, mn)
            row = {
                'Modelo': {'linear': 'Lineal', 'poly2': 'Polinomial 2', 'poly3': 'Polinomial 3'}[mn],
                'R² Ventas': f"{met_m['r2']:.4f}",
                'Ventas M28': f"${met_m['m28']:,.0f}" if met_m['m28'] else '—',
                'Ventas M30': f"${met_m['m30']:,.0f}" if met_m['m30'] else '—',
                'Prom V 28-30': f"${met_m['prom_28_30']:,.0f}" if met_m['prom_28_30'] else '—',
                'Activo': '✅' if mn == model_choice else '',
            }
            if has_contrib and met_c:
                _, met_mc = project_contrib(new_c_raw, mirror_c_all, target_months, mn)
                row['R² Contrib.'] = f"{met_mc['r2']:.4f}"
                row['Contrib M30'] = f"${met_mc['m30']:,.0f}" if met_mc['m30'] is not None else '—'
                row['Prom C 28-30'] = f"${met_mc['prom_28_30']:,.0f}" if met_mc['prom_28_30'] is not None else '—'
            summary.append(row)
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        fig_comp = go.Figure()
        colors_map = {'linear': '#555', 'poly2': '#ED1C24', 'poly3': '#FFD100'}
        for mn, color in colors_map.items():
            pr_v, _ = project_series(new_v_raw, mirror_v_all, target_months, mn)
            ponly = pr_v[pr_v['Tipo'] == 'Proyectado']
            fig_comp.add_trace(go.Scatter(
                x=ponly['Mes'], y=ponly['Valor'],
                name=f'Ventas {mn}', line=dict(color=color, width=3 if mn == model_choice else 1.5,
                                               dash='solid' if mn == model_choice else 'dash')))

        ronly = proj_v_df[proj_v_df['Tipo'] == 'Real']
        fig_comp.add_trace(go.Scatter(
            x=ronly['Mes'], y=ronly['Valor'],
            name='Real', line=dict(color='white', width=3),
            mode='lines+markers'))
        fig_comp.update_layout(
            title='Comparación de Modelos — Ventas Operativas', height=380,
            xaxis_title='Mes', yaxis_title='Ventas ($)',
            **PLOTLY_LAYOUT)
        st.plotly_chart(fig_comp, use_container_width=True)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0D0D0D,#1A0000);
border:1px solid rgba(237,28,36,0.2);border-radius:12px;
padding:1.5rem 2rem;text-align:center;'>
  <span style='font-family:Bebas Neue,sans-serif;font-size:1.5rem;
  color:#FFD100;letter-spacing:3px;'>OXXO ESPEJO</span>
  <span style='color:#555;margin:0 1rem;'>·</span>
  <span style='font-size:0.82rem;color:#666;'>
  Espejo ≥{min_months}m · Prom. últ.3m ≥${min_ventas_ult3:,.0f} · 
  Sin mes 1 · Regresión {model_choice} · Distancia euclidiana ponderada
  </span>
</div>
""", unsafe_allow_html=True)
