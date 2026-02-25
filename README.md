# 🏪 Proyección de Ventas – Tienda Espejo OXXO

App Streamlit que detecta tiendas con **< 10 meses** de operación,
encuentra su **tienda espejo más similar** (≥ 18 meses) y proyecta ventas
hasta el mes 30, destacando el **promedio de meses 28, 29 y 30**.

---

## 🚀 Deploy en GitHub + Streamlit Cloud

### Archivos del repositorio
```
mi-repo/
├── app_proyeccion.py   ← la app principal
├── requirements.txt    ← dependencias
├── data.xlsx           ← tu archivo de datos (2 hojas)
└── README.md
```

### Paso a paso

**1. Crear repositorio en GitHub**
- Ve a github.com → New repository
- Nombre: `oxxo-proyeccion` (o el que quieras)
- Visibilidad: Public o Private

**2. Subir archivos**
- Sube `app_proyeccion.py`, `requirements.txt`, `README.md`
- Sube tu `data.xlsx` (con las 2 hojas: `data` y `Hoja1`)

**3. Conectar Streamlit Cloud**
- Ve a [share.streamlit.io](https://share.streamlit.io)
- New app → selecciona tu repo
- Main file path: `app_proyeccion.py`
- Deploy ✅

---

## 📊 Estructura de data.xlsx

### Hoja: `data` — Ventas mensuales
| CR    | Tienda            | Mes A   | Ventas 6 Months |
|-------|-------------------|---------|-----------------|
| 5002I | Paris Gaitan (Bog)| jul-25  | 155421          |
| 5002I | Paris Gaitan (Bog)| ago-25  | 166089          |

### Hoja: `Hoja1` — Características tiendas
| CR    | NAME | ZONA   | MUN    | ESTRATO | TIPO DE LOCAL | AREA  | SEG26 | RENTA | GENERADOR | VT  | ET  | VU6M   | TRU6  |
|-------|------|--------|--------|---------|---------------|-------|-------|-------|-----------|-----|-----|--------|-------|
| 50WHX | 140  | Centro | Bogotá | 5       | Esquinero     | 177.4 | BASE  | 11384 | BAJA...   | 163 |3034 | 215961 | 20199 |

---

## ⚙️ Cómo funciona el modelo

1. **Detección automática** de tiendas con < 10 meses en hoja `data`
2. **Búsqueda de espejo** por distancia euclidiana ponderada en hoja `Hoja1`:
   - Filtro obligatorio por segmento (SEG26)
   - Excluye la misma tienda
   - Solo candidatos con ≥ 18 meses de operación
3. **Proyección** con regresión polinomial grado 2:
   - Aprende la curva de crecimiento del espejo
   - Factor de escala = ratio de ventas reales vs espejo en meses solapados
   - Proyecta hasta el mes 30
4. **Resultado clave**: promedio proyectado meses 28, 29 y 30

---

## 💻 Ejecución local
```bash
pip install -r requirements.txt
streamlit run app_proyeccion.py
```
