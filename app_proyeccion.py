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

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏪 Proyección de Ventas – Tienda Espejo OXXO</h1>
    <p>Proyecta ventas de tiendas nuevas (&lt;10 meses) usando su tienda espejo más similar (≥18 meses)</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# DATA LOADING — lee las 2 hojas del mismo Excel
# Hoja "data"   → historial de ventas mensuales
# Hoja "Hoja1"  → características de cada tienda (espejo)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    xls = pd.ExcelFile(file_bytes)

    # ── Ventas (hoja "data") ──────────────────────────────
    df_v = xls.parse('data')
    df_v.columns = [str(c).strip() for c in df_v.columns]

    col_cr     = next((c for c in df_v.columns if c.upper() == 'CR'), df_v.columns[0])
    col_tienda = next((c for c in df_v.columns if 'TIENDA' in c.upper() or c.upper() in ['NAME','STORE']), df_v.columns[1])
    col_mes    = next((c for c in df_v.columns if 'MES' in c.upper() or 'FECHA' in c.upper() or 'DATE' in c.upper()), df_v.columns[2])
    col_venta  = next((c for c in df_v.columns if 'VENTA' in c.upper() or 'SALES' in c.upper() or 'MONTH' in c.upper()), df_v.columns[3])

    df_v = df_v.rename(columns={col_cr:'CR', col_tienda:'Tienda', col_mes:'Mes_Raw', col_venta:'Ventas'})
    df_v['CR']       = df_v['CR'].astype(str).str.strip()
    df_v['Ventas']   = pd.to_numeric(df_v['Ventas'], errors='coerce')
    df_v['Mes_Date'] = pd.to_datetime(df_v['Mes_Raw'], errors='coerce')
    df_v = df_v.dropna(subset=['Ventas']).sort_values(['CR','Mes_Date'])
    df_v['Mes_Num']  = df_v.groupby('CR').cumcount() + 1

    # ── Espejo (hoja "Hoja1") ─────────────────────────────
    df_e = xls.parse('Hoja1')
    df_e.columns = [str(c).strip().upper() for c in df_e.columns]
    # Normalizar columna RENTA (a veces tiene espacio al final)
    df_e = df_e.rename(columns={c: 'RENTA' for c in df_e.columns if 'RENTA' in c})
    if 'VT' in df_e.columns:
        df_e['VIVIENDAS'] = df_e['VT']
    if 'ET' in df_e.columns:
        df_e['EMPLEOS'] = df_e['ET']
    for col in ['VU6M','TRU6','VIVIENDAS','EMPLEOS','AREA','ESTRATO']:
        if col not in df_e.columns:
            df_e[col] = 0
    df_e['CR'] = df_e['CR'].astype(str).str.strip()

    return df_v, df_e


# ─────────────────────────────────────────────────────────
# MIRROR STORE — distancia euclidiana ponderada
# ─────────────────────────────────────────────────────────
def find_mirror(df_espejo, nueva, df_ventas, exclude_cr, min_months, pesos):
    df_f = df_espejo[df_espejo['SEG26'] == nueva['SEG26']].copy()
    df_f = df_f[df_f['CR'] != exclude_cr]

    # Filtrar por meses mínimos de operación
    meses_cr = df_ventas.groupby('CR')['Mes_Num'].max().reset_index()
    meses_cr.columns = ['CR','total_meses']
    df_f = df_f.merge(meses_cr, on='CR', how='left')
    df_f['total_meses'] = df_f['total_meses'].fillna(0)
    df_f = df_f[df_f['total_meses'] >= min_months]

    if df_f.empty:
        return None, f"Sin tiendas con ≥{min_months} meses en segmento '{nueva['SEG26']}'"

    # Variables numéricas — normalizar con StandardScaler
    vars_num = ['ESTRATO','AREA','VIVIENDAS','EMPLEOS','VU6M','TRU6']
    X_num    = df_f[vars_num].fillna(0).values
    X_new    = np.array([[nueva.get(v, 0) for v in vars_num]])
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X_num)
    Xn_sc    = scaler.transform(X_new)

    # Variables categóricas — binarias (1 = coincide)
    vars_cat = ['ZONA','TIPO DE LOCAL','GENERADOR','MUN']
    X_cat    = np.zeros((len(df_f), len(vars_cat)))
    for i, var in enumerate(vars_cat):
        if var in df_f.columns:
            X_cat[:, i] = (df_f[var].astype(str) == str(nueva.get(var,''))).astype(int)
    Xn_cat = np.ones((1, len(vars_cat)))

    # Combinar y ponderar
    X_full  = np.hstack([X_sc,  X_cat])
    Xn_full = np.hstack([Xn_sc, Xn_cat])
    pw = np.sqrt(np.array([pesos.get(k, 0) for k in
                            ['ESTRATO','AREA','VIVIENDAS','EMPLEOS','VU6M','TRU6',
                             'ZONA','TIPO DE LOCAL','GENERADOR','MUN']]))

    dist = euclidean_distances(Xn_full * pw, X_full * pw)[0]
    mn, mx = dist.min(), dist.max()
    sim  = (1 - (dist - mn) / (mx - mn + 1e-9)) * 100

    df_f = df_f.copy()
    df_f['DISTANCIA'] = dist
    df_f['SIMILITUD'] = sim
    return df_f.sort_values('SIMILITUD', ascending=False), None


# ─────────────────────────────────────────────────────────
# PROJECTION — patrón del espejo × factor de escala
# ─────────────────────────────────────────────────────────
def project_sales(new_sales, mirror_sales, target=30, model_name='poly2'):
    X_m  = np.arange(1, len(mirror_sales)+1).reshape(-1,1)
    y_m  = np.array(mirror_sales)
    models = {
        'linear': LinearRegression().fit(X_m, y_m),
        'poly2' : Pipeline([('p',PolynomialFeatures(2)),('r',LinearRegression())]).fit(X_m, y_m),
        'poly3' : Pipeline([('p',PolynomialFeatures(3)),('r',LinearRegression())]).fit(X_m, y_m),
    }
    r2s = {n: r2_score(y_m, m.predict(X_m)) for n, m in models.items()}

    n_ov  = min(len(new_sales), len(mirror_sales))
    scale = (np.mean(new_sales[:n_ov]) / (np.mean(mirror_sales[:n_ov]) + 1e-9)) if n_ov >= 2 \
            else (new_sales[0] / (mirror_sales[0] + 1e-9))

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
        'prom_28_30': np.mean([proj[i] for i in [27,28,29] if i < target]),
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
    📋 Un solo Excel con <strong>2 hojas</strong>:<br>
    • <code>data</code> &nbsp;→ ventas mensuales<br>
    • <code>Hoja1</code> → características tiendas
    </div>
    """, unsafe_allow_html=True)

    uploaded   = st.file_uploader("Sube tu data.xlsx", type=['xlsx','xls'])
    usar_local = False
    if uploaded is None and os.path.exists('data.xlsx'):
        usar_local = st.checkbox("Usar data.xlsx del repositorio", value=True)

    st.divider()
    st.header("⚙️ Parámetros")
    min_months    = st.slider("Meses mínimos espejo",   12, 24, 18)
    target_months = st.slider("Meses a proyectar",      24, 36, 30)
    model_choice  = st.selectbox("Modelo de proyección",
                                  ['poly2','poly3','linear'],
                                  format_func=lambda x: {
                                      'poly2' :'Polinomial grado 2 ✅ (recomendado)',
                                      'poly3' :'Polinomial grado 3',
                                      'linear':'Lineal'}[x])
    st.divider()
    st.header("⚖️ Pesos Tienda Espejo")
    p = {
        'VU6M'         : st.slider("💰 Ventas U6M",    0,100,12),
        'TRU6'         : st.slider("🚶 Tráfico U6M",   0,100,10),
        'ZONA'         : st.slider("Zona",             0,100,10),
        'ESTRATO'      : st.slider("Estrato",          0,100, 8),
        'AREA'         : st.slider("Área m²",          0,100, 8),
        'TIPO DE LOCAL': st.slider("Tipo de Local",    0,100, 7),
        'GENERADOR'    : st.slider("Generador",        0,100, 7),
        'VIVIENDAS'    : st.slider("Viviendas (VT)",   0,100, 6),
        'EMPLEOS'      : st.slider("Empleos (ET)",     0,100, 6),
        'MUN'          : st.slider("Municipio",        0,100, 6),
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
    with open('data.xlsx','rb') as f:
        data_bytes = f.read()

if data_bytes is None:
    st.markdown("""
    <div class='highlight'>
    <h3>👈 Carga tu archivo Excel para comenzar</h3>
    <p>El archivo <strong>data.xlsx</strong> debe contener exactamente <strong>2 hojas</strong>:</p>
    <table>
    <tr><th>Hoja</th><th>Columnas requeridas</th></tr>
    <tr><td><code>data</code></td>
        <td><code>CR | Tienda | Mes A | Ventas 6 Months</code></td></tr>
    <tr><td><code>Hoja1</code></td>
        <td><code>CR | NAME | ZONA | MUN | ESTRATO | TIPO DE LOCAL | AREA | SEG26 | RENTA | GENERADOR | VT | ET | VU6M | TRU6</code></td></tr>
    </table>
    <br>
    <b>Opción rápida:</b> sube tu <code>data.xlsx</code> al repositorio de GitHub junto con la app
    y se cargará automáticamente sin que el usuario tenga que subirlo.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Cargando datos..."):
    df_ventas, df_espejo = load_data(data_bytes)

# Calcular meses por tienda
meses_cr  = df_ventas.groupby('CR')['Mes_Num'].max().reset_index()
meses_cr.columns = ['CR','total_meses']
nombres_cr = df_ventas.groupby('CR')['Tienda'].first().reset_index()

tiendas_nuevas = (meses_cr[meses_cr['total_meses'] < 10]
                  .merge(nombres_cr, on='CR', how='left')
                  .sort_values('total_meses'))

# Banner resumen
c1,c2,c3,c4 = st.columns(4)
c1.metric("Tiendas en base ventas",    df_ventas['CR'].nunique())
c2.metric("Tiendas en base espejo",    df_espejo['CR'].nunique())
c3.metric("Tiendas < 10 meses",        len(tiendas_nuevas))
c4.metric("Candidatos a espejo",       len(meses_cr[meses_cr['total_meses'] >= min_months]))
st.markdown("---")


# ─────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    st.subheader("🏪 Seleccionar Tienda Nueva")

    if tiendas_nuevas.empty:
        st.warning("No hay tiendas con < 10 meses en la base de ventas.")
        st.stop()

    tiendas_nuevas['label'] = tiendas_nuevas.apply(
        lambda r: f"{r['CR']} – {r['Tienda']}  ({int(r['total_meses'])} "
                  f"{'mes' if r['total_meses']==1 else 'meses'})", axis=1)

    sel_label = st.selectbox("Tiendas con < 10 meses:", tiendas_nuevas['label'].tolist())
    sel_row   = tiendas_nuevas[tiendas_nuevas['label'] == sel_label].iloc[0]
    sel_cr    = sel_row['CR']
    sel_meses = int(sel_row['total_meses'])

    mc1, mc2 = st.columns(2)
    mc1.metric("Meses reales",      sel_meses)
    mc2.metric("Meses a proyectar", target_months - sel_meses)

    # Características precargadas desde Hoja1
    row_esp = df_espejo[df_espejo['CR'] == sel_cr]
    has_esp = len(row_esp) > 0

    def get_val(col, default=0):
        if has_esp and col in row_esp.columns:
            v = row_esp.iloc[0][col]
            return default if pd.isna(v) else v
        return default

    def safe_idx(lst, val):
        try:    return list(lst).index(val)
        except: return 0

    all_segs  = sorted(df_espejo['SEG26'].dropna().unique())
    all_zonas = sorted(df_espejo['ZONA'].dropna().unique())
    all_muns  = sorted(df_espejo['MUN'].dropna().unique())
    all_tipos = sorted(df_espejo['TIPO DE LOCAL'].dropna().unique())
    all_gens  = sorted(df_espejo['GENERADOR'].dropna().unique())
    all_estr  = sorted(df_espejo['ESTRATO'].dropna().unique())

    st.markdown("**Características para buscar espejo:**")
    st.caption("Precargadas automáticamente desde Hoja1")

    seg26    = st.selectbox("Segmento (SEG26)", all_segs,  index=safe_idx(all_segs,  get_val('SEG26','BASE')))
    zona     = st.selectbox("Zona",             all_zonas, index=safe_idx(all_zonas, get_val('ZONA','')))
    mun      = st.selectbox("Municipio",        all_muns,  index=safe_idx(all_muns,  get_val('MUN','')))
    estrato  = st.selectbox("Estrato",          all_estr,  index=safe_idx(all_estr,  get_val('ESTRATO',3)))
    tipo_loc = st.selectbox("Tipo de Local",    all_tipos, index=safe_idx(all_tipos, get_val('TIPO DE LOCAL','')))
    generador= st.selectbox("Generador",        all_gens,  index=safe_idx(all_gens,  get_val('GENERADOR','')))

    ca, cb = st.columns(2)
    with ca:
        area      = st.number_input("Área (m²)",      value=float(get_val('AREA',100)),  step=5.0)
        viviendas = st.number_input("Viviendas (VT)", value=int(get_val('VT',1000)),     step=100)
        empleos   = st.number_input("Empleos (ET)",   value=int(get_val('ET',500)),      step=50)
    with cb:
        vu6m_inp  = st.number_input("Ventas U6M ($)", value=float(get_val('VU6M',0)),    step=5000.0)
        tru6_inp  = st.number_input("Tráfico U6M",    value=float(get_val('TRU6',0)),    step=500.0)

    nueva_dict = {
        'CR':sel_cr, 'SEG26':seg26, 'ZONA':zona, 'MUN':mun,
        'ESTRATO':estrato, 'TIPO DE LOCAL':tipo_loc, 'GENERADOR':generador,
        'AREA':area, 'VIVIENDAS':viviendas, 'EMPLEOS':empleos,
        'VU6M':vu6m_inp, 'TRU6':tru6_inp,
    }

    run = st.button("🚀  Proyectar Ventas", use_container_width=True)


# ─────────────────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("📊 Resultados")

    if not run:
        st.markdown("""
        <div class='highlight'>
        <b>ℹ️ Cómo funciona:</b>
        <ol>
        <li>Selecciona una tienda con &lt;10 meses de operación</li>
        <li>Verifica sus características (se pre-llenan desde tu Excel)</li>
        <li>Pulsa <b>🚀 Proyectar Ventas</b></li>
        </ol>
        El sistema buscará la tienda más similar con ≥{min} meses, aprenderá
        su curva de crecimiento y proyectará hasta el mes {target},
        mostrando los meses <b>28, 29 y 30</b> con su promedio.
        </div>
        """.replace('{min}', str(min_months)).replace('{target}', str(target_months)),
        unsafe_allow_html=True)

        st.markdown("### Tiendas disponibles (< 10 meses)")
        disp_nuevas = tiendas_nuevas[['CR','Tienda','total_meses']].rename(columns={'total_meses':'Meses'})
        st.dataframe(disp_nuevas, use_container_width=True, hide_index=True, height=340)
        st.stop()

    # ── Buscar espejo ──
    with st.spinner("Buscando tienda espejo..."):
        res_espejo, err = find_mirror(
            df_espejo, nueva_dict, df_ventas,
            exclude_cr=sel_cr, min_months=min_months, pesos=pesos)

    if err:
        st.error(f"❌ {err}")
        st.stop()

    mejor       = res_espejo.iloc[0]
    espejo_cr   = mejor['CR']
    espejo_name = str(mejor.get('NAME', espejo_cr))
    espejo_meses= int(mejor.get('total_meses', 0))

    new_sales    = df_ventas[df_ventas['CR']==sel_cr   ].sort_values('Mes_Num')['Ventas'].tolist()
    mirror_sales = df_ventas[df_ventas['CR']==espejo_cr].sort_values('Mes_Num')['Ventas'].tolist()

    proj_df, metrics = project_sales(new_sales, mirror_sales, target_months, model_choice)

    # ── Banner espejo ──
    st.success(
        f"✅ Tienda Espejo: **{espejo_name}** ({espejo_cr})  |  "
        f"Similitud: **{mejor['SIMILITUD']:.1f}%**  |  {espejo_meses} meses de operación")

    kc1,kc2,kc3,kc4 = st.columns(4)
    kc1.metric("Factor de escala",   f"{metrics['scale']:.3f}×")
    kc2.metric("R² del modelo",      f"{metrics['r2']:.4f}")
    kc3.metric("Meses espejo",       espejo_meses)
    kc4.metric("Meses reales nueva", sel_meses)

    # ── KPIs 28-30 ──
    st.markdown("---")
    st.markdown("### 🎯 Ventas Proyectadas – Meses 28, 29 y 30")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("📅 Mes 28",     f"${metrics['m28']:,.0f}"        if metrics['m28']      else "—")
    k2.metric("📅 Mes 29",     f"${metrics['m29']:,.0f}"        if metrics['m29']      else "—")
    k3.metric("📅 Mes 30",     f"${metrics['m30']:,.0f}"        if metrics['m30']      else "—")
    k4.metric("📊 Prom 28–30", f"${metrics['prom_28_30']:,.0f}" if metrics['prom_28_30'] else "—",
              help="Promedio aritmético de los meses 28, 29 y 30")

    # ── Gráfico principal ──
    st.markdown("---")
    real_d = proj_df[proj_df['Tipo']=='Real']
    proy_d = proj_df[proj_df['Tipo']=='Proyectado']
    mir_df = pd.DataFrame({'Mes': range(1, len(mirror_sales)+1), 'Ventas': mirror_sales})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mir_df['Mes'], y=mir_df['Ventas'],
        name=f'Espejo: {espejo_name}',
        line=dict(color='#FFD100', width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=3)))
    fig.add_trace(go.Scatter(
        x=real_d['Mes'], y=real_d['Ventas'],
        name=f'Nueva: {sel_cr} (real)',
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
        xaxis_title='Mes de Operación', yaxis_title='Ventas ($)',
        height=430,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📋 Tabla Proyección", "🏪 Tiendas Espejo", "📊 Comparar Modelos"])

    with tab1:
        disp = proj_df.copy()
        disp['Ventas ($)'] = disp['Ventas'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(disp[['Mes','Ventas ($)','Tipo']], use_container_width=True,
                     height=280, hide_index=True)
        csv = proj_df.to_csv(index=False)
        st.download_button("📥 Descargar CSV", csv, f"proyeccion_{sel_cr}.csv", "text/csv")

    with tab2:
        cols_show = [c for c in ['CR','NAME','ZONA','MUN','ESTRATO','TIPO DE LOCAL',
                                  'AREA','SEG26','VU6M','TRU6','SIMILITUD','total_meses']
                     if c in res_espejo.columns]
        top10 = res_espejo.head(10)[cols_show].copy()
        if 'SIMILITUD' in top10.columns:
            top10['SIMILITUD'] = top10['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
        if 'VU6M' in top10.columns:
            top10['VU6M'] = top10['VU6M'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top10.rename(columns={'total_meses':'Meses op.'}),
                     use_container_width=True, hide_index=True)

        fig_sim = px.bar(res_espejo.head(10), x='CR', y='SIMILITUD',
                          title='Top 10 por Similitud',
                          color='SIMILITUD',
                          color_continuous_scale=['#C41E3A','#ED1C24','#FFD100','#28a745'],
                          hover_data=['NAME'] if 'NAME' in res_espejo.columns else [])
        fig_sim.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)', height=320)
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab3:
        summary = []
        for mn in ['linear','poly2','poly3']:
            _, met_m = project_sales(new_sales, mirror_sales, target_months, mn)
            summary.append({
                'Modelo'    : {'linear':'Lineal','poly2':'Polinomial 2','poly3':'Polinomial 3'}[mn],
                'R²'        : f"{met_m['r2']:.4f}",
                'Mes 28 ($)': f"${met_m['m28']:,.0f}"       if met_m['m28']      else '—',
                'Mes 29 ($)': f"${met_m['m29']:,.0f}"       if met_m['m29']      else '—',
                'Mes 30 ($)': f"${met_m['m30']:,.0f}"       if met_m['m30']      else '—',
                'Prom 28-30': f"${met_m['prom_28_30']:,.0f}" if met_m['prom_28_30'] else '—',
                'Activo'    : '✅' if mn == model_choice else '',
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        fig_c = go.Figure()
        colors_map = {'linear':'#aaaaaa','poly2':'#ED1C24','poly3':'#FFD100'}
        for mn, color in colors_map.items():
            pr, _ = project_sales(new_sales, mirror_sales, target_months, mn)
            ponly = pr[pr['Tipo']=='Proyectado']
            fig_c.add_trace(go.Scatter(
                x=ponly['Mes'], y=ponly['Ventas'], name=mn,
                line=dict(color=color,
                          width=3 if mn==model_choice else 1.5,
                          dash='solid' if mn==model_choice else 'dash')))
        ronly = proj_df[proj_df['Tipo']=='Real']
        fig_c.add_trace(go.Scatter(
            x=ronly['Mes'], y=ronly['Ventas'],
            name='Real', line=dict(color='#1a1a1a',width=3),
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
    Distancia euclidiana ponderada + Regresión polinomial con factor de escala</p>
</div>
""", unsafe_allow_html=True)
