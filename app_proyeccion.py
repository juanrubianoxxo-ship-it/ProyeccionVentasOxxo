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
    page_title="Proyección de Ventas OXXO",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #ED1C24 0%, #C41E3A 100%);
    padding: 2rem; border-radius: 10px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.15);
}
.main-header h1 { color: white; font-size: 2rem; font-weight: bold; margin: 0; }
.main-header p  { color: #FFD100; font-size: 1rem; margin: 0.4rem 0 0 0; }
.stButton>button {
    background-color: #ED1C24 !important; color: white !important;
    border: none; border-radius: 6px; padding: 0.6rem 2rem;
    font-weight: bold; font-size: 1rem; transition: all 0.3s;
}
.stButton>button:hover { background-color: #C41E3A !important; transform: translateY(-2px); }
[data-testid="stMetricValue"] { color: #ED1C24; font-weight: bold; }
[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFD100 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p   { color: #ffffff !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background-color: #f0f0f0; border-radius: 5px 5px 0 0; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background-color: #ED1C24 !important; color: white !important; }
.stDownloadButton>button { background-color: #FFD100 !important; color: #1a1a1a !important; font-weight: bold; }
.highlight { background: #fff9e6; border-left: 4px solid #FFD100;
             border-radius: 6px; padding: 1rem; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-header'>
    <h1>🏪 Proyección de Ventas – Tienda Espejo OXXO</h1>
    <p>Proyecta ventas de tiendas nuevas (&lt;10 meses) usando su tienda espejo más similar (≥30 meses)</p>
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
    except:
        pass
    return pd.NaT


# ─────────────────────────────────────────────────────────
# DATA LOADING  — LibroVentas.xlsx
#   Hoja "INFO TIENDAS"     → características de cada tienda
#   Hoja "HISTORICO VENTAS" → tabla pivote: CR en filas, mes en columnas
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    xls = pd.ExcelFile(file_bytes)

    # ── INFO TIENDAS ──────────────────────────────────────
    df_info = xls.parse('INFO TIENDAS')
    df_info.columns = [str(c).strip().upper() for c in df_info.columns]
    df_info = df_info.rename(columns={c: 'RENTA' for c in df_info.columns if 'RENTA' in c})
    if 'VT' in df_info.columns:
        df_info['VIVIENDAS'] = df_info['VT']
    if 'ET' in df_info.columns:
        df_info['EMPLEOS'] = df_info['ET']
    for col in ['VIVIENDAS','EMPLEOS','AREA','ESTRATO']:
        if col not in df_info.columns:
            df_info[col] = 0
    if 'SEG26' not in df_info.columns:
        df_info['SEG26'] = 'BASE'
    df_info['CR'] = df_info['CR'].astype(str).str.strip()

    # ── HISTORICO VENTAS (tabla pivote exportada desde Excel) ─
    # Estructura: fila 4 = encabezados fecha | filas 5+ = datos CR
    df_raw = xls.parse('HISTORICO VENTAS', header=None)
    header_row  = df_raw.iloc[5, :]
    date_labels = header_row[1:].tolist()

    df_piv = df_raw.iloc[6:, :].copy()
    df_piv.columns = ['CR'] + date_labels
    df_piv = df_piv.reset_index(drop=True)
    df_piv['CR'] = df_piv['CR'].astype(str).str.strip()

    if 'Total general' in df_piv.columns:
        df_piv = df_piv.drop(columns=['Total general'])

    df_v = df_piv.melt(id_vars=['CR'], var_name='Fecha_Raw', value_name='Ventas')
    df_v = df_v.dropna(subset=['Ventas'])
    df_v['Ventas'] = pd.to_numeric(df_v['Ventas'], errors='coerce')
    df_v = df_v.dropna(subset=['Ventas'])
    df_v['Fecha'] = df_v['Fecha_Raw'].apply(parse_fecha)
    df_v = df_v.dropna(subset=['Fecha'])
    df_v = df_v.sort_values(['CR','Fecha']).reset_index(drop=True)
    df_v['Mes_Num'] = df_v.groupby('CR').cumcount() + 1

    return df_v, df_info


# ─────────────────────────────────────────────────────────
# MIRROR STORE — distancia euclidiana ponderada
#   Candidatos: >= min_months meses  Y  ventas_prom >= min_ventas
# ─────────────────────────────────────────────────────────
def find_mirror(df_info, nueva, df_ventas, exclude_cr, min_months, pesos, min_ventas):
    df_f = df_info[df_info['SEG26'] == nueva['SEG26']].copy()
    df_f = df_f[df_f['CR'] != exclude_cr]

    meses_cr = (df_ventas.groupby('CR')['Mes_Num'].max()
                .reset_index().rename(columns={'Mes_Num':'total_meses'}))
    df_f = df_f.merge(meses_cr, on='CR', how='left')
    df_f['total_meses'] = df_f['total_meses'].fillna(0)
    df_f = df_f[df_f['total_meses'] >= min_months]

    if df_f.empty:
        return None, f"Sin tiendas con ≥{min_months} meses en segmento '{nueva['SEG26']}'"

    ventas_prom = (df_ventas.groupby('CR')['Ventas'].mean()
                   .reset_index().rename(columns={'Ventas':'ventas_prom'}))
    df_f = df_f.merge(ventas_prom, on='CR', how='left')
    df_f['ventas_prom'] = df_f['ventas_prom'].fillna(0)

    n_antes = len(df_f)
    df_f = df_f[df_f['ventas_prom'] >= min_ventas]
    if df_f.empty:
        return None, (
            f"Sin tiendas espejo con ventas prom ≥ ${min_ventas:,.0f} "
            f"en segmento '{nueva['SEG26']}' con ≥{min_months} meses. "
            f"({n_antes} candidatas antes del filtro de ventas)")

    vars_num = ['ESTRATO','AREA','VIVIENDAS','EMPLEOS']
    X_num  = df_f[vars_num].fillna(0).values
    X_new  = np.array([[nueva.get(v,0) for v in vars_num]])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_num)
    Xn_sc  = scaler.transform(X_new)

    vars_cat = ['ZONA','TIPO DE LOCAL','GENERADOR','MUN']
    X_cat = np.zeros((len(df_f), len(vars_cat)))
    for i, var in enumerate(vars_cat):
        if var in df_f.columns:
            X_cat[:, i] = (df_f[var].astype(str) == str(nueva.get(var,''))).astype(int)
    Xn_cat = np.ones((1, len(vars_cat)))

    X_full  = np.hstack([X_sc,  X_cat])
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
    return df_f.sort_values('SIMILITUD', ascending=False), None


# ─────────────────────────────────────────────────────────
# PROJECTION
#   ▸ Espejo  → solo sus PRIMEROS 30 meses para entrenar
#   ▸ Nueva   → se DESCARTA el mes 1 registrado
# ─────────────────────────────────────────────────────────
def project_sales(new_sales_raw, mirror_sales_all, target=30, model_name='poly2'):
    # Espejo: primeros 30 meses
    mirror_sales = mirror_sales_all[:30]

    # Nueva: sin el primer mes
    new_sales = new_sales_raw[1:] if len(new_sales_raw) > 1 else new_sales_raw

    X_m = np.arange(1, len(mirror_sales)+1).reshape(-1,1)
    y_m = np.array(mirror_sales, dtype=float)

    models = {
        'linear': LinearRegression().fit(X_m, y_m),
        'poly2' : Pipeline([('p',PolynomialFeatures(2)),('r',LinearRegression())]).fit(X_m, y_m),
        'poly3' : Pipeline([('p',PolynomialFeatures(3)),('r',LinearRegression())]).fit(X_m, y_m),
    }
    r2s = {n: r2_score(y_m, m.predict(X_m)) for n, m in models.items()}

    n_ov  = min(len(new_sales), len(mirror_sales))
    scale = (np.mean(new_sales[:n_ov]) / (np.mean(mirror_sales[:n_ov]) + 1e-9)) if n_ov >= 2 \
            else (new_sales[0] / (mirror_sales[0] + 1e-9) if new_sales else 1.0)

    X_all = np.arange(1, target+1).reshape(-1,1)
    proj  = np.clip(models[model_name].predict(X_all) * scale, 0, None)
    cur   = len(new_sales)

    df_res = pd.DataFrame({
        'Mes'   : range(1, target+1),
        'Ventas': [new_sales[i] if i < cur else proj[i] for i in range(target)],
        'Tipo'  : ['Real' if i < cur else 'Proyectado' for i in range(target)],
    })
    metrics = {
        'scale': scale, 'r2': r2s[model_name], 'r2s': r2s,
        'm28': proj[27] if target >= 28 else None,
        'm29': proj[28] if target >= 29 else None,
        'm30': proj[29] if target >= 30 else None,
        'prom_28_30':       np.mean([proj[i] for i in [27,28,29] if i < target]),
        'meses_reales_usados': cur,
        'meses_espejo_usados': len(mirror_sales),
    }
    return df_res, metrics


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Archivo de Datos")
    st.markdown("""
    <div style='background:#ED1C24;padding:0.7rem;border-radius:6px;
                color:white;font-size:0.85rem;line-height:1.5;'>
    📋 Excel con <strong>2 hojas</strong>:<br>
    • <code>INFO TIENDAS</code> → características<br>
    • <code>HISTORICO VENTAS</code> → tabla pivote mensual
    </div>
    """, unsafe_allow_html=True)

    uploaded   = st.file_uploader("Sube tu LibroVentas.xlsx", type=['xlsx','xls'])
    usar_local = False
    if uploaded is None and os.path.exists('LibroVentas.xlsx'):
        usar_local = st.checkbox("Usar LibroVentas.xlsx del repositorio", value=True)

    st.divider()
    st.header("⚙️ Parámetros")
    min_months    = st.slider("Meses mínimos espejo",  20, 40, 30,
                               help="Tiendas con menos meses no se consideran como espejo")
    target_months = st.slider("Meses a proyectar",     24, 36, 30)
    model_choice  = st.selectbox("Modelo de proyección",
                                  ['poly2','poly3','linear'],
                                  format_func=lambda x: {
                                      'poly2' :'Polinomial grado 2 ✅ (recomendado)',
                                      'poly3' :'Polinomial grado 3',
                                      'linear':'Lineal'}[x])

    st.divider()
    st.header("💰 Filtro Ventas Espejo")
    min_ventas_espejo = st.number_input(
        "Ventas promedio mínimas ($)",
        min_value=0, value=200_000, step=10_000,
        help="Solo se consideran espejos cuyas ventas promedio superen este umbral"
    )
    st.caption(f"Umbral activo: **${min_ventas_espejo:,.0f}**")

    st.markdown("""
    <div style='background:#333;padding:0.6rem;border-radius:6px;
                color:#FFD100;font-size:0.82rem;line-height:1.6;margin-top:0.5rem;'>
    📌 <b>Reglas de proyección:</b><br>
    • Espejo → se usan sus <b>primeros 30 meses</b><br>
    • Nueva &nbsp;→ se <b>descarta el mes 1</b> registrado
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("⚖️ Pesos Tienda Espejo")
    p = {
        'ZONA'         : st.slider("Zona",             0,100,15),
        'ESTRATO'      : st.slider("Estrato",          0,100,12),
        'AREA'         : st.slider("Área m²",          0,100,12),
        'TIPO DE LOCAL': st.slider("Tipo de Local",    0,100,12),
        'GENERADOR'    : st.slider("Generador",        0,100,10),
        'VIVIENDAS'    : st.slider("Viviendas (VT)",   0,100, 8),
        'EMPLEOS'      : st.slider("Empleos (ET)",     0,100, 8),
        'MUN'          : st.slider("Municipio",        0,100, 8),
    }
    total_p = sum(p.values()) or 1
    pesos   = {k: v/total_p*0.70 for k,v in p.items()}
    pesos['SEG26'] = 0.30

    with st.expander("Ver pesos normalizados"):
        for k, v in sorted(pesos.items(), key=lambda x: -x[1]):
            st.write(f"**{k}:** {v*100:.1f}%")


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
data_bytes = None
if uploaded:
    data_bytes = uploaded.read()
elif usar_local:
    with open('LibroVentas.xlsx','rb') as f:
        data_bytes = f.read()

if data_bytes is None:
    st.markdown("""
    <div class='highlight'>
    <h3>👈 Carga tu archivo Excel para comenzar</h3>
    <p>El archivo debe contener exactamente <strong>2 hojas</strong>:</p>
    <table>
    <tr><th>Hoja</th><th>Descripción</th></tr>
    <tr><td><code>INFO TIENDAS</code></td>
        <td>CR, NAME, ZONA, MUN, ESTRATO, TIPO DE LOCAL, AREA, SEG26, RENTA, GENERADOR, VT, ET</td></tr>
    <tr><td><code>HISTORICO VENTAS</code></td>
        <td>Tabla pivote con CR en filas y meses (ene 17, feb 17…) en columnas</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Cargando datos..."):
    df_ventas, df_info = load_data(data_bytes)

# ── Estadísticas resumen ──────────────────────────────────
meses_cr        = df_ventas.groupby('CR')['Mes_Num'].max().reset_index()
meses_cr.columns = ['CR','total_meses']
ventas_prom_all = df_ventas.groupby('CR')['Ventas'].mean()

tiendas_nuevas = meses_cr[meses_cr['total_meses'] < 10].copy()
tiendas_nuevas = tiendas_nuevas.merge(
    df_info[['CR','NAME']].drop_duplicates(), on='CR', how='left')

candidatos_espejo = meses_cr[
    (meses_cr['total_meses'] >= min_months) &
    (meses_cr['CR'].map(ventas_prom_all).fillna(0) >= min_ventas_espejo)
]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Tiendas totales",                         df_ventas['CR'].nunique())
c2.metric("Tiendas en INFO",                         df_info['CR'].nunique())
c3.metric("Nuevas (< 10 meses)",                     len(tiendas_nuevas))
c4.metric(f"Candidatos espejo (≥${min_ventas_espejo/1000:.0f}K)", len(candidatos_espejo))
st.markdown("---")


# ─────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    st.subheader("🏪 Seleccionar Tienda Nueva")

    if tiendas_nuevas.empty:
        st.warning("No hay tiendas con < 10 meses en el histórico.")
        st.stop()

    tiendas_nuevas['label'] = tiendas_nuevas.apply(
        lambda r: f"{r['CR']}  —  {r.get('NAME','?')}  ({int(r['total_meses'])} "
                  f"{'mes' if r['total_meses']==1 else 'meses'})", axis=1)

    sel_label = st.selectbox("Tiendas con < 10 meses:", tiendas_nuevas['label'].tolist())
    sel_row   = tiendas_nuevas[tiendas_nuevas['label'] == sel_label].iloc[0]
    sel_cr    = sel_row['CR']
    sel_meses_total  = int(sel_row['total_meses'])
    sel_meses_usados = max(0, sel_meses_total - 1)   # sin mes 1

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Meses registrados",     sel_meses_total)
    mc2.metric("Meses usados (sin M1)", sel_meses_usados)
    mc3.metric("Meses a proyectar",     target_months - sel_meses_usados)

    # Características precargadas desde INFO TIENDAS
    row_info = df_info[df_info['CR'] == sel_cr]
    has_info = len(row_info) > 0

    def get_val(col, default=0):
        if has_info and col in row_info.columns:
            v = row_info.iloc[0][col]
            return default if pd.isna(v) else v
        return default

    def safe_idx(lst, val):
        try:    return list(lst).index(val)
        except: return 0

    all_segs  = sorted(df_info['SEG26'].dropna().unique())
    all_zonas = sorted(df_info['ZONA'].dropna().unique())
    all_muns  = sorted(df_info['MUN'].dropna().unique())
    all_tipos = sorted(df_info['TIPO DE LOCAL'].dropna().unique())
    all_gens  = sorted(df_info['GENERADOR'].dropna().unique())
    all_estr  = sorted(df_info['ESTRATO'].dropna().unique())

    st.markdown("**Características para buscar espejo:**")
    st.caption("Pre-cargadas desde INFO TIENDAS")

    seg26    = st.selectbox("Segmento (SEG26)", all_segs,  index=safe_idx(all_segs,  get_val('SEG26','BASE')))
    zona     = st.selectbox("Zona",             all_zonas, index=safe_idx(all_zonas, get_val('ZONA','')))
    mun      = st.selectbox("Municipio",        all_muns,  index=safe_idx(all_muns,  get_val('MUN','')))
    estrato  = st.selectbox("Estrato",          all_estr,  index=safe_idx(all_estr,  get_val('ESTRATO',3)))
    tipo_loc = st.selectbox("Tipo de Local",    all_tipos, index=safe_idx(all_tipos, get_val('TIPO DE LOCAL','')))
    generador= st.selectbox("Generador",        all_gens,  index=safe_idx(all_gens,  get_val('GENERADOR','')))

    ca, cb = st.columns(2)
    with ca:
        area      = st.number_input("Área (m²)",      value=float(get_val('AREA',100)), step=5.0)
        viviendas = st.number_input("Viviendas (VT)", value=int(get_val('VT',1000)),    step=100)
    with cb:
        empleos   = st.number_input("Empleos (ET)",   value=int(get_val('ET',500)),     step=50)

    nueva_dict = {
        'CR':sel_cr, 'SEG26':seg26, 'ZONA':zona, 'MUN':mun,
        'ESTRATO':estrato, 'TIPO DE LOCAL':tipo_loc, 'GENERADOR':generador,
        'AREA':area, 'VIVIENDAS':viviendas, 'EMPLEOS':empleos,
    }

    run = st.button("🚀  Proyectar Ventas", use_container_width=True)


# ─────────────────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("📊 Resultados")

    if not run:
        st.markdown(f"""
        <div class='highlight'>
        <b>ℹ️ Cómo funciona esta versión:</b>
        <ol>
        <li>Selecciona una tienda nueva (&lt;10 meses)</li>
        <li>Verifica sus características (auto-cargadas desde <em>INFO TIENDAS</em>)</li>
        <li>Pulsa <b>🚀 Proyectar Ventas</b></li>
        </ol>
        <b>Reglas aplicadas:</b><br>
        🔹 <b>Espejo:</b> ≥{min_months} meses y ventas prom ≥ ${min_ventas_espejo:,.0f} —
           se usan solo sus <b>primeros 30 meses</b> para entrenar el modelo<br>
        🔹 <b>Nueva:</b> se <b>descarta el mes 1</b> registrado (apertura atípica)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Tiendas disponibles (< 10 meses)")
        disp = tiendas_nuevas[['CR','NAME','total_meses']].rename(
            columns={'total_meses':'Meses registrados','NAME':'Nombre'})
        st.dataframe(disp, use_container_width=True, hide_index=True, height=340)
        st.stop()

    # ── Buscar espejo ──────────────────────────────────────
    with st.spinner("Buscando tienda espejo..."):
        res_espejo, err = find_mirror(
            df_info, nueva_dict, df_ventas,
            exclude_cr=sel_cr, min_months=min_months,
            pesos=pesos, min_ventas=min_ventas_espejo)

    if err:
        st.error(f"❌ {err}")
        st.stop()

    mejor        = res_espejo.iloc[0]
    espejo_cr    = mejor['CR']
    espejo_name  = str(mejor.get('NAME', espejo_cr))
    espejo_meses = int(mejor.get('total_meses', 0))
    espejo_vprom = mejor.get('ventas_prom', 0)

    new_sales_raw    = df_ventas[df_ventas['CR']==sel_cr   ].sort_values('Mes_Num')['Ventas'].tolist()
    mirror_sales_all = df_ventas[df_ventas['CR']==espejo_cr].sort_values('Mes_Num')['Ventas'].tolist()

    proj_df, metrics = project_sales(new_sales_raw, mirror_sales_all, target_months, model_choice)

    # ── Banner espejo ──────────────────────────────────────
    st.success(
        f"✅ Tienda Espejo: **{espejo_name}** ({espejo_cr})  |  "
        f"Similitud: **{mejor['SIMILITUD']:.1f}%**  |  "
        f"Meses totales: {espejo_meses} → se usaron primeros **{metrics['meses_espejo_usados']}**  |  "
        f"Venta prom: **${espejo_vprom:,.0f}**")

    kc1,kc2,kc3,kc4,kc5 = st.columns(5)
    kc1.metric("Factor escala",          f"{metrics['scale']:.3f}×")
    kc2.metric("R² modelo",              f"{metrics['r2']:.4f}")
    kc3.metric("Meses espejo usados",    metrics['meses_espejo_usados'])
    kc4.metric("Meses reales nueva",     metrics['meses_reales_usados'],
               help="Meses efectivos de la tienda nueva (descartado M1)")
    kc5.metric("Venta prom espejo",      f"${espejo_vprom:,.0f}")

    # ── KPIs 28–30 ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Ventas Proyectadas – Meses 28, 29 y 30")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("📅 Mes 28",     f"${metrics['m28']:,.0f}"         if metrics['m28']        else "—")
    k2.metric("📅 Mes 29",     f"${metrics['m29']:,.0f}"         if metrics['m29']        else "—")
    k3.metric("📅 Mes 30",     f"${metrics['m30']:,.0f}"         if metrics['m30']        else "—")
    k4.metric("📊 Prom 28–30", f"${metrics['prom_28_30']:,.0f}"  if metrics['prom_28_30'] else "—",
              help="Promedio aritmético de los meses 28, 29 y 30")

    # ── Gráfico principal ──────────────────────────────────
    st.markdown("---")
    real_d      = proj_df[proj_df['Tipo']=='Real']
    proy_d      = proj_df[proj_df['Tipo']=='Proyectado']
    mirror_plot = mirror_sales_all[:30]
    mir_df      = pd.DataFrame({'Mes': range(1, len(mirror_plot)+1), 'Ventas': mirror_plot})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mir_df['Mes'], y=mir_df['Ventas'],
        name=f'Espejo: {espejo_name} (M1–30)',
        line=dict(color='#FFD100', width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=3)))
    fig.add_trace(go.Scatter(
        x=real_d['Mes'], y=real_d['Ventas'],
        name=f'{sel_cr} (real, sin M1)',
        line=dict(color='#ED1C24', width=3),
        mode='lines+markers', marker=dict(size=6)))
    fig.add_trace(go.Scatter(
        x=proy_d['Mes'], y=proy_d['Ventas'],
        name='Proyección',
        line=dict(color='#C41E3A', width=2, dash='dash'),
        fill='tozeroy', fillcolor='rgba(237,28,36,0.06)'))

    for mes, val in [(28,metrics['m28']),(29,metrics['m29']),(30,metrics['m30'])]:
        if val and target_months >= mes:
            fig.add_vline(x=mes, line_color='#FFD100', line_dash='dot', opacity=0.6)
            fig.add_annotation(x=mes, y=val,
                text=f"M{mes}<br>${val/1000:.0f}K",
                showarrow=True, arrowhead=2,
                bgcolor='#FFD100', font=dict(color='#1a1a1a', size=11), borderpad=4)

    fig.update_layout(
        title=f'Proyección de Ventas – {sel_cr}  (modelo: {model_choice})',
        xaxis_title='Mes de Operación (M1 = 2do mes real de la tienda nueva)',
        yaxis_title='Ventas ($)', height=430,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Tabla Proyección","🏪 Tiendas Espejo","📊 Comparar Modelos"])

    with tab1:
        disp_t = proj_df.copy()
        disp_t['Ventas ($)'] = disp_t['Ventas'].apply(lambda x: f"${x:,.0f}")
        st.caption("⚠️ M1 aquí = segundo mes real registrado de la tienda (M1 original descartado)")
        st.dataframe(disp_t[['Mes','Ventas ($)','Tipo']],
                     use_container_width=True, height=300, hide_index=True)
        csv = proj_df.to_csv(index=False)
        st.download_button("📥 Descargar CSV", csv, f"proyeccion_{sel_cr}.csv", "text/csv")

    with tab2:
        cols_show = [c for c in ['CR','NAME','ZONA','MUN','ESTRATO','TIPO DE LOCAL',
                                  'AREA','SEG26','ventas_prom','SIMILITUD','total_meses']
                     if c in res_espejo.columns]
        top10 = res_espejo.head(10)[cols_show].copy()
        if 'SIMILITUD'   in top10.columns: top10['SIMILITUD']   = top10['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
        if 'ventas_prom' in top10.columns: top10['ventas_prom'] = top10['ventas_prom'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top10.rename(columns={'total_meses':'Meses tot.','ventas_prom':'Venta Prom'}),
                     use_container_width=True, hide_index=True)

        fig_sim = px.bar(res_espejo.head(10), x='CR', y='SIMILITUD',
                          title=f'Top 10 candidatos espejo (ventas prom ≥ ${min_ventas_espejo:,.0f})',
                          color='SIMILITUD',
                          color_continuous_scale=['#C41E3A','#ED1C24','#FFD100','#28a745'],
                          hover_data=['NAME','ventas_prom'] if 'NAME' in res_espejo.columns else ['ventas_prom'])
        fig_sim.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab3:
        summary = []
        for mn in ['linear','poly2','poly3']:
            _, met_m = project_sales(new_sales_raw, mirror_sales_all, target_months, mn)
            summary.append({
                'Modelo'    : {'linear':'Lineal','poly2':'Polinomial 2','poly3':'Polinomial 3'}[mn],
                'R²'        : f"{met_m['r2']:.4f}",
                'Mes 28 ($)': f"${met_m['m28']:,.0f}"        if met_m['m28']        else '—',
                'Mes 29 ($)': f"${met_m['m29']:,.0f}"        if met_m['m29']        else '—',
                'Mes 30 ($)': f"${met_m['m30']:,.0f}"        if met_m['m30']        else '—',
                'Prom 28-30': f"${met_m['prom_28_30']:,.0f}" if met_m['prom_28_30'] else '—',
                'Activo'    : '✅' if mn == model_choice else '',
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        fig_c = go.Figure()
        colors_map = {'linear':'#aaaaaa','poly2':'#ED1C24','poly3':'#FFD100'}
        for mn, color in colors_map.items():
            pr, _ = project_sales(new_sales_raw, mirror_sales_all, target_months, mn)
            ponly = pr[pr['Tipo']=='Proyectado']
            fig_c.add_trace(go.Scatter(
                x=ponly['Mes'], y=ponly['Ventas'], name=mn,
                line=dict(color=color,
                          width=3 if mn==model_choice else 1.5,
                          dash='solid' if mn==model_choice else 'dash')))
        ronly = proj_df[proj_df['Tipo']=='Real']
        fig_c.add_trace(go.Scatter(
            x=ronly['Mes'], y=ronly['Ventas'],
            name='Real (sin M1)', line=dict(color='#1a1a1a', width=3),
            mode='lines+markers'))
        fig_c.update_layout(
            title='Comparación de Modelos', height=380,
            xaxis_title='Mes', yaxis_title='Ventas ($)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;padding:1.2rem;
     background:linear-gradient(90deg,#ED1C24,#C41E3A);border-radius:10px;'>
    <h4 style='color:#FFD100;margin:0;'>🏪 Proyección de Ventas – Tienda Espejo OXXO</h4>
    <p style='color:white;margin:0.3rem 0 0;font-size:0.85rem;'>
    Espejo: primeros 30 meses · Nueva: sin mes 1 · Distancia euclidiana + Regresión polinomial</p>
</div>
""", unsafe_allow_html=True)
