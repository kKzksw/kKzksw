import os
from pathlib import Path
import pandas as pd
import numpy as np
import dill
from PIL import Image
import streamlit as st

from catboost import CatBoostClassifier

from eclyon.transforms import process_df


path_to_repo = Path(__file__).parent.resolve()
path_to_data = path_to_repo /'dataset'/'creditcard_processed.csv'
path_to_model = path_to_repo /'machine learning'/'machine learning'/'catboost_model.pk'



def display_credit_score(index):
    pred_class = st.session_state.model.predict([st.session_state.X.iloc[index]])[0]
    pred_proba = st.session_state.model.predict_proba([st.session_state.X.iloc[index]])[0]
    actu_class = st.session_state.y[index]

    st.subheader('Credit Score Prediction')
    st.write(f'Predicted Credit Score: {pred_class}')
    st.write(f'Probability of Prediction: {max(pred_proba):.2%}')
    st.write(f'Actual Score: {actu_class}')


def display_customer_features(index):
    st.subheader('Customer Details')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.iloc[index]
    for i, feature in enumerate(st.session_state.X.columns):
        if i % 2 == 0:
            with feat0:
                st.info(feature)
            with val0:
                st.success(str(row[i]))
        else:
            with feat1:
                st.info(feature)
            with val1:
                st.success(str(row[i]))
                
def init_session_state():
    if 'loaded' not in st.session_state:
        df_raw = pd.read_csv(path_to_data)
        X, y, nas = process_df(df_raw, 'credit_score')
        
        path_to_model = os.path.join(path_to_repo, 'catboost_model.pk')
        
        with open(path_to_model, 'rb') as file:
            model = dill.load(file)

        st.session_state.loaded = True
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.model = model



def app():
    init_session_state()
    st.title('Bank Credit Scoring Prediction')
    options = ['-'] + list(range(1, len(st.session_state.X) + 1))
    index = st.selectbox(label='Choose a Customer Index', options=options, index=0)
    if index != '-':
        index = int(index) - 1
        display_credit_score(index)
        display_customer_features(index)


if __name__ == '__main__':
    app()