import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="RUL Compresor", layout="wide")

st.title("🔧 APP RUL COMPRESOR - RANDOM FOREST")

# 📥 CARGA DE DATOS
@st.cache_data
def load_data():
    df = pd.read_excel("interdiciplinario7_preparacion_RUL(4).xlsx",
                       sheet_name="Datos_Preparados_RUL")
    return df

df = load_data()

st.subheader("📊 Datos")
st.dataframe(df.head())

# VARIABLES
target = "RUL_limpio_h"

features = [
    'norm_MTTF','norm_MTTR','norm_tasa_fallos','norm_indice_fiabilidad',
    'norm_temperatura_operacion','norm_corriente_entrada',
    'norm_frecuencia_operacion','norm_temperatura_proceso',
    'norm_presion_proceso','norm_consumo_energia',
    'norm_tiempo_inactividad','norm_numero_fallos',
    'norm_nivel_obsolescencia','norm_vida_util_componentes'
]

df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELO
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# MÉTRICAS
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Métricas")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R2: {r2:.3f}")

# GRÁFICA
st.subheader("📉 RUL Real vs Predicho")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
st.pyplot(fig)

# PREDICCIÓN
st.subheader("🔮 Predicción")

input_data = {}
for f in features:
    input_data[f] = st.number_input(f, value=float(df[f].mean()))

if st.button("Predecir RUL"):
    pred = model.predict(pd.DataFrame([input_data]))[0]
    st.success(f"RUL estimado: {pred:.2f} horas")

# GUARDAR MODELO
if st.button("Guardar modelo"):
    joblib.dump(model, "modelo_rul_random_forest.joblib")
    st.success("Modelo guardado")