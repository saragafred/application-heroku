import streamlit as st
from PIL import Image

import numpy as np
import pandas as pd
from data_api import *
import time
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.figure import Figure
import streamlit.components.v1 as components
from urllib.request import urlopen
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from urllib.request import urlopen
import plotly.graph_objects as go
import json

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Loetitia Rabier">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Green; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:Orange; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)


# Cacher le bouton en haut à droite
st.markdown(""" <style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================
sample_size = 20000
data ,train_set,y_pred_test_export = load_all_data(sample_size)

#--------------------------- Client Predection --------------------------

def show_client_predection():
    client_id = st.number_input("Donnez Id du Client",100002)
    if st.button('Voir Client'):
        client=data[data['SK_ID_CURR']==client_id]
        
        display_client_info(str(client['SK_ID_CURR'].values[0]),str(client['AMT_INCOME_TOTAL'].values[0]),str(round(client['DAYS_BIRTH'].values[0])),str(round(client['DAYS_EMPLOYED']/-365).values[0]))
        
        
        #st.header('ID :'+str(client['SK_ID_CURR'][0]))
        #st.write(data['age_bins'].value_counts())
        API_url = "https://appflask2023.herokuapp.com/credit/" + str(client_id)
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            y_pred = API_data['prediction']
            y_proba = API_data['proba']
            
        
        
        
        
        st.info('Prediction du client : '+str(int(100*y_proba))+' %')
        client_prediction= st.progress(0)
        for percent_complete in range(int(100*y_proba)):
            time.sleep(0.01)

        client_prediction.progress(percent_complete + 1)
        if(y_proba < 0.5):
            st.success('Client solvable')
        if(y_proba >=0.5):
            st.error('Client non solvable')

        st.subheader("Tous les détails du client :")
        st.write(client)
    
    

#--------------------------- model analysis -------------------------
### Confusion matrixe
def matrix_confusion (X,y):
    cm = confusion_matrix(X, y)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
    return  cm

def show_model_analysis():
    conf_mtx = matrix_confusion(y_pred_test_export['y_test'],y_pred_test_export['y_predicted'])
    #st.write(conf_mtx)
    fig = go.Figure(data=go.Heatmap(
                   z=conf_mtx,
                    x=[ 'Actual Negative:0','Actual Positive:1'],
                   y=['Predict Negative:0','Predict Positive:1'],
                   hoverongaps = False))
    st.plotly_chart(fig)

    fpr, tpr, thresholds = roc_curve(y_pred_test_export['y_test'],y_pred_test_export['y_probability'])

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)
    
    
    
    
    
    

# ====================================================================
# IMAGES
# ====================================================================
# Logo de l'entreprise                  
logo =  Image.open("./depense.jpg")


# --------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')




st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Menu:',
    ['Model & Prediction'],
)
    
    
if sidebar_selection == 'Model & Prediction':
    selected_item = st.sidebar.selectbox('Select Menu:', 
                                    ( 'Prediction','Model'))

    

if selected_item == 'Prediction':
    show_client_predection()

if selected_item == 'Model':
    show_model_analysis()


