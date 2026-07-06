"""
Telecom Customer Churn Prediction App
--------------------------------------
A professional Streamlit application that serves a trained Decision Tree
Pipeline (StandardScaler + DecisionTreeClassifier) to predict whether a
telecom customer is likely to churn.

Author: Senior ML Engineer
"""

import io
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# CONSTANTS — must mirror the feature engineering done during training
# --------------------------------------------------------------------------------
MODEL_PATH = "best_model.pkl"

STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
    "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
]
# "AK" was the reference/baseline category dropped by pd.get_dummies(drop_first=True)
DUMMY_STATES = [s for s in STATES if s != "AK"]

# Revenue_Segment tertile boundaries learned from the training data via pd.qcut
# (Low: -inf–55.05 | Medium: 55.05–63.91 | High: 63.91–inf)
REVENUE_BIN_EDGES = [-np.inf, 55.05, 63.913333, np.inf]
REVENUE_LABELS = ["Low", "Medium", "High"]

# The exact column order the trained pipeline expects
FEATURE_ORDER = [
    "Account length", "International plan", "Voice mail plan",
    "Total day minutes", "Total day calls", "Total eve minutes",
    "Total eve calls", "Total night minutes", "Total night calls",
    "Total intl minutes", "Total intl calls", "Customer service calls",
    "Total Charges", "Total_Usage", "Service_Stress",
] + [f"State_{s}" for s in DUMMY_STATES] + [
    "Revenue_Segment_Medium", "Revenue_Segment_High",
]


# --------------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .app-header {
            padding: 1.75rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #4338CA 0%, #7C3AED 50%, #C026D3 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(88, 28, 135, 0.25);
        }
        .app-header h1 {
            margin: 0;
            font-weight: 800;
            font-size: 2rem;
        }
        .app-header p {
            margin: 0.4rem 0 0 0;
            opacity: 0.92;
            font-size: 0.98rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #EEF0F5;
            border-radius: 14px;
            padding: 1.25rem 1.4rem 0.6rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .section-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #312E81;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.7rem 1rem;
            background: linear-gradient(135deg, #4338CA 0%, #7C3AED 100%);
            color: white;
            border: none;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            border: 1.5px solid #4338CA;
            color: #4338CA;
            background: white;
        }
        div.stDownloadButton > button:hover {
            background: #F5F3FF;
        }

        .result-churn {
            background: linear-gradient(135deg, #FEE2E2, #FFF1F2);
            border: 1px solid #FCA5A5;
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            text-align: center;
        }
        .result-safe {
            background: linear-gradient(135deg, #DCFCE7, #F0FDF4);
            border: 1px solid #86EFAC;
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            text-align: center;
        }
        .result-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 0.2rem; }
        .result-sub { font-size: 0.95rem; opacity: 0.85; }

        .footer-note {
            text-align: center;
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model pipeline…")
def load_model(path: str):
    return joblib.load(path)


# --------------------------------------------------------------------------------
# FEATURE ENGINEERING (mirrors the training notebook exactly)
# --------------------------------------------------------------------------------
def build_feature_row(inputs: dict) -> pd.DataFrame:
    total_charges = (
        inputs["day_charge"] + inputs["eve_charge"]
        + inputs["night_charge"] + inputs["intl_charge"]
    )
    total_usage = (
        inputs["day_minutes"] + inputs["eve_minutes"]
        + inputs["night_minutes"] + inputs["intl_minutes"]
    )
    service_stress = inputs["service_calls"] / (inputs["account_length"] + 1)

    revenue_segment = pd.cut(
        [total_charges], bins=REVENUE_BIN_EDGES, labels=REVENUE_LABELS
    )[0]

    row = {
        "Account length": inputs["account_length"],
        "International plan": 1 if inputs["intl_plan"] == "Yes" else 0,
        "Voice mail plan": 1 if inputs["vmail_plan"] == "Yes" else 0,
        "Total day minutes": inputs["day_minutes"],
        "Total day calls": inputs["day_calls"],
        "Total eve minutes": inputs["eve_minutes"],
        "Total eve calls": inputs["eve_calls"],
        "Total night minutes": inputs["night_minutes"],
        "Total night calls": inputs["night_calls"],
        "Total intl minutes": inputs["intl_minutes"],
        "Total intl calls": inputs["intl_calls"],
        "Customer service calls": inputs["service_calls"],
        "Total Charges": total_charges,
        "Total_Usage": total_usage,
        "Service_Stress": service_stress,
    }

    for s in DUMMY_STATES:
        row[f"State_{s}"] = 1 if inputs["state"] == s else 0

    row["Revenue_Segment_Medium"] = 1 if revenue_segment == "Medium" else 0
    row["Revenue_Segment_High"] = 1 if revenue_segment == "High" else 0

    df = pd.DataFrame([row])
    df = df[FEATURE_ORDER]  # enforce exact training column order
    return df, total_charges, total_usage, service_stress, revenue_segment


# --------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📡 Churn Predictor")
        st.caption("Telecom Customer Retention Toolkit")
        st.divider()

        st.markdown("### 🧠 Model Information")
        st.markdown(
            "- **Algorithm:** Decision Tree Classifier\n"
            "- **Pipeline:** StandardScaler → DecisionTreeClassifier\n"
            "- **Criterion:** Entropy\n"
            "- **Max depth:** 5\n"
            "- **Features used:** 67"
        )
        st.divider()

        st.markdown("### ℹ️ How to use")
        st.markdown(
            "1. Fill in the customer's account & usage details.\n"
            "2. Click **Predict Churn**.\n"
            "3. Review the churn probability and risk verdict.\n"
            "4. Download the result as a CSV record."
        )
        st.divider()

        st.markdown("### ⚙️ Session")
        if "history" not in st.session_state:
            st.session_state.history = []
        st.metric("Predictions this session", len(st.session_state.history))
        if st.session_state.history:
            if st.button("🗑️ Clear session history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.divider()
        st.caption("Built with Streamlit · scikit-learn pipeline")


# --------------------------------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------------------------------
def render_form():
    st.markdown(
        """
        <div class="app-header">
            <h1>📡 Telecom Customer Churn Prediction</h1>
            <p>Enter a customer's account and usage profile to estimate their likelihood of churning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">👤 Account Profile</div>', unsafe_allow_html=True)
            state = st.selectbox("State", STATES, index=STATES.index("KS"))
            account_length = st.number_input("Account length (days)", min_value=0, max_value=400, value=100)
            intl_plan = st.radio("International plan", ["No", "Yes"], horizontal=True)
            vmail_plan = st.radio("Voice mail plan", ["No", "Yes"], horizontal=True)
            service_calls = st.number_input("Customer service calls", min_value=0, max_value=20, value=1)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">☀️ Day &amp; 🌆 Evening Usage</div>', unsafe_allow_html=True)
            day_minutes = st.number_input("Total day minutes", min_value=0.0, max_value=400.0, value=180.0, step=0.1)
            day_calls = st.number_input("Total day calls", min_value=0, max_value=200, value=100)
            day_charge = st.number_input("Total day charge ($)", min_value=0.0, max_value=100.0, value=30.6, step=0.01)
            eve_minutes = st.number_input("Total eve minutes", min_value=0.0, max_value=400.0, value=200.0, step=0.1)
            eve_calls = st.number_input("Total eve calls", min_value=0, max_value=200, value=100)
            eve_charge = st.number_input("Total eve charge ($)", min_value=0.0, max_value=50.0, value=17.0, step=0.01)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🌙 Night &amp; 🌍 International</div>', unsafe_allow_html=True)
            night_minutes = st.number_input("Total night minutes", min_value=0.0, max_value=400.0, value=200.0, step=0.1)
            night_calls = st.number_input("Total night calls", min_value=0, max_value=200, value=100)
            night_charge = st.number_input("Total night charge ($)", min_value=0.0, max_value=30.0, value=9.0, step=0.01)
            intl_minutes = st.number_input("Total intl minutes", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
            intl_calls = st.number_input("Total intl calls", min_value=0, max_value=20, value=4)
            intl_charge = st.number_input("Total intl charge ($)", min_value=0.0, max_value=10.0, value=2.7, step=0.01)
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict Churn")

    inputs = dict(
        state=state, account_length=account_length, intl_plan=intl_plan,
        vmail_plan=vmail_plan, service_calls=service_calls,
        day_minutes=day_minutes, day_calls=day_calls, day_charge=day_charge,
        eve_minutes=eve_minutes, eve_calls=eve_calls, eve_charge=eve_charge,
        night_minutes=night_minutes, night_calls=night_calls, night_charge=night_charge,
        intl_minutes=intl_minutes, intl_calls=intl_calls, intl_charge=intl_charge,
    )
    return submitted, inputs


# --------------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------------
def render_result(inputs, model):
    try:
        feature_row, total_charges, total_usage, service_stress, revenue_segment = build_feature_row(inputs)

        # Defensive validation before inference
        if feature_row.isnull().values.any():
            st.error("⚠️ Some computed features are missing (NaN). Please check your inputs and try again.")
            return

        if feature_row.shape[1] != len(model.feature_names_in_):
            st.error(
                f"⚠️ Feature mismatch: built {feature_row.shape[1]} features but the model "
                f"expects {len(model.feature_names_in_)}. Please contact the app maintainer."
            )
            return

        pred = model.predict(feature_row)[0]
        proba = model.predict_proba(feature_row)[0]
        churn_prob = proba[1]
        stay_prob = proba[0]

    except Exception as e:
        st.error(f"🚫 Prediction failed due to an unexpected error: `{e}`")
        st.info("Double-check that all fields are filled with realistic numeric values, then try again.")
        return

    st.markdown("### 🎯 Prediction Result")
    res_col, gauge_col = st.columns([1.1, 1])

    with res_col:
        if pred == 1:
            st.markdown(
                f"""
                <div class="result-churn">
                    <div class="result-title">⚠️ High Risk — Likely to Churn</div>
                    <div class="result-sub">Churn probability: <b>{churn_prob:.1%}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-safe">
                    <div class="result-title">✅ Low Risk — Likely to Stay</div>
                    <div class="result-sub">Retention probability: <b>{stay_prob:.1%}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with gauge_col:
        st.markdown("**Prediction Probability Breakdown**")
        st.progress(float(churn_prob), text=f"Churn: {churn_prob:.1%}")
        st.progress(float(stay_prob), text=f"Stay: {stay_prob:.1%}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Charges", f"${total_charges:,.2f}")
        m2.metric("Total Usage (min)", f"{total_usage:,.1f}")
        m3.metric("Revenue Segment", revenue_segment)

    with st.expander("🔍 View engineered feature vector sent to the model"):
        st.dataframe(feature_row.T.rename(columns={0: "value"}), use_container_width=True)

    # Build a downloadable record
    record = inputs.copy()
    record.update(
        {
            "total_charges": round(total_charges, 2),
            "total_usage": round(total_usage, 2),
            "service_stress": round(service_stress, 4),
            "revenue_segment": revenue_segment,
            "predicted_churn": int(pred),
            "churn_probability": round(float(churn_prob), 4),
            "stay_probability": round(float(stay_prob), 4),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    )
    st.session_state.history.append(record)

    result_df = pd.DataFrame([record])
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download This Prediction (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"churn_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if len(st.session_state.history) > 1:
        with st.expander(f"📜 Session history ({len(st.session_state.history)} predictions)"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            hist_buffer = io.StringIO()
            hist_df.to_csv(hist_buffer, index=False)
            st.download_button(
                "⬇️ Download Full Session History (CSV)",
                data=hist_buffer.getvalue(),
                file_name="churn_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
def main():
    inject_css()
    render_sidebar()

    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"🚫 Model file not found at `{MODEL_PATH}`. Please make sure `best_model.pkl` is in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"🚫 Failed to load the model: `{e}`")
        st.stop()

    submitted, inputs = render_form()

    if submitted:
        render_result(inputs, model)
    else:
        st.info("👆 Fill in the customer details above and click **Predict Churn** to get started.")

    st.markdown(
        '<div class="footer-note">Telecom Churn Predictor · Decision Tree Pipeline · For internal analytics use</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
