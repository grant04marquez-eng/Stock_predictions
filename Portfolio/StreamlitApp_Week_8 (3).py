import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile
import json

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline

import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Access the secrets
aws_id      = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret  = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token   = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket  = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

TARGET      = 'AMZN'
RETURN_PERIOD = 5

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Load SP500 data to get realistic price bounds for AMZN
@st.cache_data
def load_sp500(_session, bucket):
    s3_client = _session.client('s3')
    local_csv  = os.path.join(tempfile.gettempdir(), 'SP500Data.csv')
    if not os.path.exists(local_csv):
        s3_client.download_file(
            Bucket=bucket,
            Key='SP500Data.csv',
            Filename=local_csv
        )
    return pd.read_csv(local_csv, index_col=0)

df_sp500  = load_sp500(session, aws_bucket)
amzn_prices = df_sp500[TARGET]

MIN_VAL     = float(amzn_prices.min() * 0.5)
MAX_VAL     = float(amzn_prices.max() * 2.0)
DEFAULT_VAL = float(amzn_prices.mean())

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pca.shap',
    "pipeline":  'finalized_pca_model.tar.gz',
    "inputs": [{"name": "AMZN Close Price", "min": MIN_VAL, "default": DEFAULT_VAL, "step": 10.0}]
}


def load_pipeline(_session, bucket, s3_key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    local_tar = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(Bucket=bucket, Key=f"{s3_key}/{filename}", Filename=local_tar)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))


def load_shap_explainer(_session, bucket, s3_key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# Prediction Logic
def call_model_api(amzn_price):
    """
    Sends the AMZN close price as JSON to the endpoint.
    The inference_pca.py input_fn handles feature reconstruction.
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    payload = json.dumps({TARGET: amzn_price})

    try:
        raw_pred = predictor.predict(payload)
        # Option 1 is regression — return the predicted cumulative return
        pred_val = float(np.array(raw_pred).flatten()[0])
        return pred_val, 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(amzn_price, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_explainer = os.path.join(tempfile.gettempdir(), explainer_name)
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        local_explainer
    )

    full_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Reconstruct the feature row locally for SHAP (same logic as inference_pca.py)
    closest_date = (df_sp500[TARGET] - amzn_price).abs().idxmin()
    X_full = np.log(df_sp500.drop([TARGET], axis=1)).diff(RETURN_PERIOD)
    X_full = np.exp(X_full).cumsum()
    X_full.columns = [name + '_CR_Cum' for name in X_full.columns]
    input_row = X_full.loc[[closest_date]]

    # Preprocessing pipeline = everything except the final model step
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-1])
    input_transformed = preprocessing_pipeline.transform(input_row)

    # KernelPCA components don't have named features — label them
    n_components = input_transformed.shape[1]
    feature_names = [f'KPC_{i+1}' for i in range(n_components)]
    input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_transformed_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)

    top_feature = pd.Series(shap_values[0].values, index=feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

with st.form("pred_form"):
    st.subheader("Inputs")
    inp = MODEL_INFO["inputs"][0]
    amzn_price = st.number_input(
        inp["name"].upper(),
        min_value=inp["min"],
        value=inp["default"],
        step=inp["step"]
    )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    res, status = call_model_api(amzn_price)
    if status == 200:
        st.metric("Predicted AMZN Cumulative Return", f"{res:.4f}")
        display_explanation(amzn_price, session, aws_bucket)
    else:
        st.error(res)
