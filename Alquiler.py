import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
import joblib
from flask import Flask, request, jsonify

model_xgb = joblib.load('C:/Users/dpino/Documents/DATA SCIENCE/Jupyter/BOOTCAMP/Mi proyecto/MODELOS/best_Xg2_model.joblib')
model_rf = joblib.load('C:/Users/dpino/Documents/DATA SCIENCE/Jupyter/BOOTCAMP/Mi proyecto/MODELOS/best_rfr2_model.joblib')
meta_model = joblib.load("C:\\Users\\dpino\\Documents\\DATA SCIENCE\\Jupyter\\BOOTCAMP\\Mi proyecto\\MODELOS\\meta_model.joblib")

# Cargar el DataFrame
df = pd.read_csv("C:\\Users\\dpino\\Documents\\DATA SCIENCE\\Jupyter\\BOOTCAMP\\Mi proyecto\\Alquiler_Precio.csv")

st.markdown("<h1 style='text-align: center;'>MI BARRIO IDEAL</h1>", unsafe_allow_html=True)
st.header('Predice tu barrio ideal y su precio, de acuerdo con tus preferencias en Barcelona')
multiseleccion = {'Camaras_seguridad': 'Presencia de camaras de seguridad',	'Transporte': 'Acceso a transporte publico', 	'Areas verdes': 'Presencia de parques y jardines', 'Equip social': 'Equipamientos sociales, educativos y deportivos',	'Equip ocio': 'Equipamiento de ocio nocturno, restaurante, museo y zoologicos'}
selected_variables = st.multiselect('Que caracteristicas debe cumplir tu zona:', multiseleccion.values())
resultado = [1 if nombre in selected_variables else 0 for nombre in multiseleccion.keys()]
min_baños = st.number_input('Cual es el minimo de baños que quieres en tu vivienda?:', min_value=1, step=1)

min_habitaciones = st.number_input('Cual es el minimo de habitaciones que quieres en tu vivienda?:', min_value=1, step=1)

min_personas = 0

min_m2 = st.slider('Cual es el minimo de m2 que quieres en tu vivienda?:', min_value= 51, max_value=113, step=1, value=51)

min_precio = st.slider('Cual es el precio minimo que quieres pagar?:', min_value= 491, max_value=1815, step=1, value=1815)

st.header('La recomendación de barrios, segun tus necesidades son:')

filtro = df[(df['Camaras_seguridad'] >= resultado[0]) & 
            (df['Transporte'] >= resultado[1]) & 
            (df['Areas verdes'] >= resultado[2]) & 
            (df['Equip social'] >= resultado[3]) & 
            (df['Equip ocio'] >= resultado[4]) & 
            (df['Med_baños'] >= min_baños) & 
            (df['Med_Hab'] >= min_habitaciones) & 
            (df['Med_Pers'] >= min_personas) & 
            (df['m2'] >= min_m2)]

if not filtro.empty:
    st.write("Los siguientes barrios cumplen con tus criterios de busqueda:")
    st.write(filtro['Nom_Barri'].tolist()[:5])
else:
    st.write("No se encontraron barrios que cumplan con los criterios de busqueda.")

columns = {
    'Camaras_seguridad': [resultado[0], resultado[0], resultado[0]],
    'Total_contratos': [df['Total_contratos'].median()] * 3,
    'm2': [min_m2] * 3,
    'Med_Hab': [df['Med_Hab'].median()] * 3,
    'Med_baños': [df['Med_baños'].median()] * 3,
    'Med_Pers': [df['Med_Pers'].median()] * 3,
    'Transporte': [resultado[1], resultado[1], resultado[1]],
    'Areas verdes': [resultado[2], resultado[2], resultado[2]],
    'Equip social': [resultado[3], resultado[3], resultado[3]],
    'Equip ocio': [resultado[4], resultado[4], resultado[4]]
}

columns_df = pd.DataFrame(columns)
df_relevante = df[columns_df.columns]

if st.button('Calcula el precio de tu recomendación'):
    if not filtro.empty:
        # Seleccionar solo los tres primeros barrios recomendados
        barrios_seleccionados = filtro['Nom_Barri'].tolist()[:5]

        # Filtrar el DataFrame original para obtener solo los datos de los tres barrios seleccionados
        df_seleccionados = df[df['Nom_Barri'].isin(barrios_seleccionados)]

        # Seleccionar solo las columnas relevantes para la predicción
        df_relevante = df_seleccionados[columns_df.columns]

        # Realizar las predicciones para los barrios seleccionados
        pred_xgb = model_xgb.predict(df_relevante)
        pred_rf = model_rf.predict(df_relevante)

        # Organizar las predicciones de los modelos base en una matriz bidimensional
        concatenated_predictions = np.column_stack((pred_xgb, pred_rf))

        # Calcular la predicción con el modelo meta
        pred_meta_model = meta_model.predict(concatenated_predictions)

        # Mostrar solo las predicciones
        st.write("El precio que deberas pagar es:")
        for i, barrio in enumerate(barrios_seleccionados):
            precio_redondeado = round(pred_meta_model[i], 2)
            st.write(f"{barrio}: {precio_redondeado} €")
    else:
        st.write("No hay barrios para tu búsqueda.")