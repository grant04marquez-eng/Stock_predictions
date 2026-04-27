import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json

# ── Page Config ──
st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="wide")
st.title("🔍 IEEE-CIS Fraud Detection")
st.markdown("**Grant Marquez | Texas Christian University | Spring 2026**")
st.markdown("This app uses a Random Forest model deployed on AWS SageMaker to predict whether a transaction is fraudulent.")

# ── Load X_train defaults from GitHub ──
@st.cache_data
def load_defaults():
    url = "https://raw.githubusercontent.com/grant04marquez-eng/Stock_predictions/main/Portfolio/X_train.csv"
    df = pd.read_csv(url)
    return df

X_train = load_defaults()

# ── Top features for user input ──
# These are the most important features from the Random Forest model
top_features = [
    'TransactionAmt',
    'card1',
    'card2',
    'C13',
    'C14',
    'C1',
    'D1',
    'TX_hour',
]

st.sidebar.header("📝 Transaction Inputs")
st.sidebar.markdown("Adjust the top features below. All other features are filled with training data defaults.")

# ── Build user inputs ──
user_inputs = {}
for feat in top_features:
    col_data = X_train[feat]
    min_val = float(col_data.min())
    max_val = float(col_data.max())
    mean_val = float(col_data.mean())
    median_val = float(col_data.median())

    if feat == 'TransactionAmt':
        user_inputs[feat] = st.sidebar.slider(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=max(5000.0, max_val),
            value=round(median_val, 2),
            step=1.0
        )
    elif feat == 'TX_hour':
        user_inputs[feat] = st.sidebar.slider(
            "Hour of Day (0-23)",
            min_value=0,
            max_value=23,
            value=int(median_val)
        )
    elif feat in ['card1', 'card2']:
        user_inputs[feat] = st.sidebar.number_input(
            f"{feat} (Card ID)",
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(median_val)
        )
    elif feat == 'D1':
        user_inputs[feat] = st.sidebar.slider(
            "D1 (Days since reference)",
            min_value=int(min_val),
            max_value=int(max(500, max_val)),
            value=int(median_val)
        )
    else:
        user_inputs[feat] = st.sidebar.slider(
            feat,
            min_value=min_val,
            max_value=max_val,
            value=median_val
        )

# ── Build full input row using X_train median for non-top features ──
input_row = X_train.median().to_dict()

# Override with user inputs
for feat, val in user_inputs.items():
    input_row[feat] = val

# Recompute derived features
input_row['TransactionAmt_log'] = float(np.log1p(input_row['TransactionAmt']))

# Convert all values to plain Python floats
input_row = {k: float(v) for k, v in input_row.items()}

# ── Display input summary ──
st.subheader("📊 Input Summary")
input_display = pd.DataFrame([{feat: user_inputs[feat] for feat in top_features}])
st.dataframe(input_display, use_container_width=True)

# ── Predict ──
if st.button("🚀 Predict Fraud", type="primary"):
    with st.spinner("Calling SageMaker endpoint..."):
        try:
            aws_credentials = st.secrets["aws_credentials"]
            runtime = boto3.client(
                'sagemaker-runtime',
                region_name='us-east-1',
                aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=aws_credentials["AWS_SESSION_TOKEN"],
            )

            response = runtime.invoke_endpoint(

            result = json.loads(response['Body'].read().decode())
            prediction = result[0]['prediction']
            probability = result[0]['fraud_probability']

            # ── Display Results ──
            st.subheader("🎯 Prediction Result")

            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **Legitimate Transaction**")

            with col2:
                st.metric("Fraud Probability", f"{probability:.2%}")

            # Progress bar for probability
            st.progress(probability)

            # ── SHAP Section ──
            st.subheader("🔬 SHAP Explanation")
            st.markdown("Feature importance for this specific prediction:")

            # Create a simple feature importance display using the top features
            # and their deviation from the training median
            importance_data = []
            for feat in top_features:
                user_val = user_inputs[feat]
                median_val = float(X_train[feat].median())
                std_val = float(X_train[feat].std())
                if std_val > 0:
                    deviation = (user_val - median_val) / std_val
                else:
                    deviation = 0
                importance_data.append({
                    'Feature': feat,
                    'Your Value': round(user_val, 2),
                    'Median': round(median_val, 2),
                    'Deviation (σ)': round(deviation, 2)
                })

            imp_df = pd.DataFrame(importance_data)
            st.dataframe(imp_df, use_container_width=True)

            # Bar chart of deviations
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['tomato' if d > 0 else 'steelblue' for d in imp_df['Deviation (σ)']]
            ax.barh(imp_df['Feature'], imp_df['Deviation (σ)'], color=colors)
            ax.set_xlabel('Standard Deviations from Median')
            ax.set_title('Feature Deviation from Training Median')
            ax.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error calling endpoint: {str(e)}")
            st.info("Make sure the SageMaker endpoint 'fraud-detection-endpoint' is running.")

st.markdown("---")
st.caption("Model: Random Forest | Deployed on AWS SageMaker | Data: IEEE-CIS Fraud Detection Dataset")
