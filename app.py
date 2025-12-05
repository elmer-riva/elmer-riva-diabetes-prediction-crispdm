import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# --- Título y Descripción ---
st.title("🩺 Sistema de Predicción de Riesgo de Diabetes")
st.markdown("""
Esta aplicación utiliza un modelo de Inteligencia Artificial (**Random Forest**) entrenado bajo la metodología CRISP-DM
para estimar la probabilidad de que un paciente padezca diabetes basándose en sus indicadores clínicos.
""")

# --- Cargar el Modelo ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('modelo_diabetes_final.joblib')
    except FileNotFoundError:
        st.error("⚠️ No se encontró el modelo. Asegúrese de ejecutar el notebook de entrenamiento primero.")
        return None

model = load_model()

# --- Sidebar: Inputs del Usuario ---
st.sidebar.header("📝 Datos del Paciente")
st.sidebar.markdown("Ingrese los valores clínicos a continuación:")

# Inputs Numéricos
age = st.sidebar.slider("Edad", 0, 100, 45)
bmi = st.sidebar.number_input("IMC (Índice de Masa Corporal)", 10.0, 96.0, 27.3)
hba1c = st.sidebar.number_input("Nivel HbA1c", 3.5, 9.0, 5.5, step=0.1, help="Nivel de hemoglobina glicosilada")
glucose = st.sidebar.number_input("Glucosa en Sangre", 80, 300, 140)

st.sidebar.markdown("---")
# Inputs Categóricos
gender = st.sidebar.selectbox("Género", ["Female", "Male"])
hypertension = st.sidebar.radio("¿Padece Hipertensión?", ["No", "Sí"], horizontal=True)
heart_disease = st.sidebar.radio("¿Padece Enfermedad Cardíaca?", ["No", "Sí"], horizontal=True)

# Historial de fumador (incluye el manejo de 'No Info')
smoking_map = {
    "Nunca (never)": "never",
    "No actual (not current)": "not current",
    "Ex-fumador (former)": "former",
    "Actual (current)": "current",
    "Alguna vez (ever)": "ever",
    "Sin Información": "Unknown"
}
smoking_input = st.sidebar.selectbox("Historial de Tabaquismo", list(smoking_map.keys()))
smoking_val = smoking_map[smoking_input]

# --- Lógica de Pestañas (Tabs) ---
tab1, tab2 = st.tabs(["🔍 Predicción Individual", "📊 Rendimiento del Modelo"])

# === PESTAÑA 1: PREDICCIÓN ===
with tab1:
    if model:
        # 1. Preprocesamiento de Datos (Recrear One-Hot Encoding)
        # Inicializamos el diccionario con todas las columnas que espera el modelo en 0
        input_data = {
            'age': age,
            'hypertension': 1 if hypertension == "Sí" else 0,
            'heart_disease': 1 if heart_disease == "Sí" else 0,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose,
            'gender_Male': 0,
            'smoking_history_current': 0,
            'smoking_history_ever': 0,
            'smoking_history_former': 0,
            'smoking_history_never': 0,
            'smoking_history_not current': 0
        }

        # Ajustamos los valores One-Hot según la selección
        if gender == "Male":
            input_data['gender_Male'] = 1

        if smoking_val != "Unknown":
            # Construimos la clave, ej: 'smoking_history_current'
            key = f"smoking_history_{smoking_val}"
            if key in input_data:
                input_data[key] = 1

        # Crear DataFrame
        df_input = pd.DataFrame([input_data])

        # Mostrar datos ingresados (feedback al usuario)
        st.info("**Datos a analizar:**")
        st.dataframe(df_input)

        col_pred, col_viz = st.columns([1, 2])

        with col_pred:
            if st.button("⚡ Calcular Riesgo", use_container_width=True):
                # Predicción
                prediction = model.predict(df_input)[0]
                probability = model.predict_proba(df_input)[0][1]

                st.divider()
                if prediction == 1:
                    st.error("### 🔴 Resultado: ALTO RIESGO")
                    st.write("El modelo sugiere una alta probabilidad de diabetes.")
                else:
                    st.success("### 🟢 Resultado: BAJO RIESGO")
                    st.write("El modelo no detecta indicadores críticos de diabetes.")

                st.metric(label="Probabilidad Estimada", value=f"{probability:.2%}")

        with col_viz:
            # Visualización simple del riesgo
            if 'probability' in locals():
                st.write("**Nivel de Riesgo:**")
                st.progress(int(probability * 100))
                if probability > 0.5:
                    st.warning("⚠️ Se recomienda consultar con un especialista para pruebas confirmatorias (Glucosa en ayunas/HbA1c).")

# === PESTAÑA 2: RENDIMIENTO DEL MODELO ===
with tab2:
    st.header("Evaluación Técnica del Modelo")
    st.write("A continuación se presentan las métricas de rendimiento obtenidas durante la fase de testeo del proyecto.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matriz de Confusión")
        try:
            img_cm = Image.open("grafico_confusion_matrix.png")
            st.image(img_cm, caption="Desempeño en clasificación de clases", use_container_width=True)
        except:
            st.warning("Imagen de Matriz de Confusión no encontrada.")

        st.subheader("Importancia de Variables")
        try:
            img_fi = Image.open("grafico_feature_importance.png")
            st.image(img_fi, caption="Variables que más influyen en la decisión del modelo", use_container_width=True)
        except:
             st.warning("Imagen de Importancia de Variables no encontrada.")

    with col_b:
        st.subheader("Curva ROC")
        try:
            img_roc = Image.open("grafico_roc_curve.png")
            st.image(img_roc, caption="Capacidad de distinción entre clases", use_container_width=True)
        except:
             st.warning("Imagen de Curva ROC no encontrada.")

        st.info("""
        **Interpretación de Variables Clave:**
        * **HbA1c_level:** Es el predictor más fuerte. Niveles altos están fuertemente correlacionados con el diagnóstico positivo.
        * **blood_glucose_level:** Segundo predictor en importancia, confirmando la relevancia de los análisis sanguíneos.
        * **BMI y Edad:** Actúan como factores de riesgo secundarios pero significativos.
        """)
