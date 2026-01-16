import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Zomato Churn Analytics (Streamlit)",
    page_icon="📊",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("app/churn_model.pkl")


# ================= TITLE =================
st.title("📊 Zomato Customer Churn Analytics")
st.caption("Interactive customer churn prediction dashboard built using Streamlit")

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Data (CSV)",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Churn Probability Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

show_only_high_risk = st.sidebar.checkbox("Show only high-risk customers")

st.sidebar.info(
    "ℹ️ Threshold Guide:\n"
    "- Lower value → catch more churn customers\n"
    "- Higher value → fewer false alerts"
)

# ================= MAIN LOGIC =================
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    required_cols = ["OrderFrequency", "DaysSinceLastOrder", "AvgRating", "Complaints"]

    if not all(col in data.columns for col in required_cols):
        st.error("CSV must contain these columns: " + ", ".join(required_cols))
    else:
        X = data[required_cols]

        # ---------------- PREDICTIONS ----------------
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        data["Churn_Probability"] = probs.round(2)
        data["Churn_Prediction"] = preds

        def recommend_action(p):
            if p >= 0.7:
                return "High Priority Retention Offer"
            elif p >= threshold:
                return "Medium Priority Retention Offer"
            else:
                return "No Immediate Action Needed"

        data["Recommended_Action"] = data["Churn_Probability"].apply(recommend_action)

        # ================= KPI SECTION =================
        st.markdown("## 📌 Key Metrics")

        k1, k2, k3, k4 = st.columns(4)

        total_customers = len(data)
        high_risk = (data["Churn_Prediction"] == 1).sum()
        low_risk = (data["Churn_Prediction"] == 0).sum()
        avg_risk = data["Churn_Probability"].mean()

        k1.metric("Total Customers", total_customers)
        k2.metric("High-Risk Customers", high_risk)
        k3.metric("Low-Risk Customers", low_risk)
        k4.metric("Average Churn Risk", f"{avg_risk:.2%}")

        # ================= VISUAL ANALYSIS =================
        st.markdown("## 📊 Visual Analysis")

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.countplot(x="Churn_Prediction", data=data, ax=ax)
            ax.set_title("Churn Prediction Distribution")
            ax.set_xlabel("Prediction (0 = Safe, 1 = Churn)")
            ax.set_ylabel("Customer Count")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            ax.hist(data["Churn_Probability"], bins=10)
            ax.axvline(threshold, linestyle="--", label="Selected Threshold")
            ax.set_title("Churn Probability Distribution")
            ax.set_xlabel("Churn Probability")
            ax.set_ylabel("Customer Count")
            ax.legend()
            st.pyplot(fig)

        # ================= CUSTOMER TABLE =================
        st.markdown("## 📋 Customer-Level Insights")

        display_data = data.copy()
        if show_only_high_risk:
            display_data = display_data[display_data["Churn_Prediction"] == 1]

        st.dataframe(display_data, use_container_width=True)

        # ================= DOWNLOAD =================
        csv = display_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

        # ================= SINGLE CUSTOMER SIMULATION =================
        st.markdown("## 🧪 Single Customer Churn Simulation")

        with st.form("single_customer_form"):
            s1, s2, s3, s4 = st.columns(4)

            of = s1.number_input("Order Frequency", 1, 50, 10)
            days = s2.number_input("Days Since Last Order", 1, 200, 30)
            rating = s3.slider("Average Rating", 1.0, 5.0, 4.0)
            comp = s4.number_input("Complaints", 0, 10, 0)

            submit = st.form_submit_button("Predict Churn Risk")

        if submit:
            single_df = pd.DataFrame({
                "OrderFrequency": [of],
                "DaysSinceLastOrder": [days],
                "AvgRating": [rating],
                "Complaints": [comp]
            })

            prob = model.predict_proba(single_df)[:, 1][0]

            st.metric("Predicted Churn Probability", f"{prob:.2%}")

            if prob >= threshold:
                st.error("⚠️ High Churn Risk – Retention Action Recommended")
            else:
                st.success("✅ Low Churn Risk – Customer Appears Stable")

else:
    st.info("👈 Upload a CSV file using the sidebar to begin analysis")
