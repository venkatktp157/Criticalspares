#!/usr/bin/env python
# coding: utf-8

# In[2]:

import streamlit as st
from scipy.stats import poisson
import matplotlib.pyplot as plt
from auth import load_authenticator
from logger import setup_logger

# === Authentication ===
authenticator = load_authenticator()
name, auth_status, username = authenticator.login('Login', 'main')
logger = setup_logger()

if auth_status:
    # === Password Reset Check ===
    if authenticator.credentials["usernames"][username].get("password_reset", False):
        st.warning("🔒 You are required to reset your password.")
        if st.button("Change Password"):
            authenticator.reset_password(username)
            st.success("✅ Password updated. Please log in again.")
            st.stop()
    
    authenticator.logout('Logout', 'main')

    st.title("📈 Time Series Anomaly Detection with SHAP")
    st.write(f"Welcome *{name}* 👋")
    logger.info(f"User {username} logged in successfully")


    # === Poisson Probability Function ===
    def poisson_probability(x, lambda_value):
        return poisson.pmf(x, lambda_value)

    st.title("📦⚙️ Critical Spares Evaluation")

    # Input Fields
    A = st.number_input("Enter A - Units needing spares per vessel (EA):", min_value=0, step=1)
    N = st.number_input("Enter N - Number of vessels:", min_value=1, step=1)
    M = st.number_input("Enter M - Running hours per vessel per month:", min_value=0, step=25)
    T = st.number_input("Enter T - Time in months:", min_value=0, step=1)
    MTBR = st.number_input("Enter MTBR - Mean Time Between Repair:", min_value=0, step=250)

    if MTBR == 0:
        st.error("MTBR cannot be zero.")
    else:
        probability_threshold = st.slider("Select Desired Probability Threshold:", 0.0, 0.99, 0.9)
        spare_part_cost = st.number_input("Enter Cost per Spare Part ($):", min_value=1, step=50)

        # === λ Calculation ===
        λ = (A * N * M * T) / MTBR
        st.markdown(f"""
        <div style='text-align:center; padding:12px;'>
        <span style='font-size:26px; color:green;'>Calculated λ (lambda): <strong>{λ:.4f}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        # 🧠 Lambda Insight
        total_runtime_hours = M * T  # Per vessel over time
        fleet_hours = total_runtime_hours * N
        total_unit_hours = fleet_hours * A

        insight_text = f"""
        <div style='padding:10px; background-color:#fff8e1; border-left: 5px solid #ffb300; margin-top:10px; font-size:16px;'>
        🧠 <strong>Insight:</strong> Over <strong>{fleet_hours:,} hours</strong> of vessel operation,
        with <strong>{A}</strong> spare-eligible units and MTBR of <strong>{MTBR}</strong>, 
        you're statistically expecting around <strong>{λ:.2f} failures</strong> across the system.
        </div>
        """

        st.markdown(insight_text, unsafe_allow_html=True)

        cumulative_probability = 0
        x = 0
        probability_table = []
        max_iterations = 1000

        while x <= max_iterations:
            p = poisson_probability(x, λ)
            cumulative_probability += p
            probability_table.append({
                "Number of Spares": x,
                "Probability": round(p, 4),
                "Cumulative Probability": round(cumulative_probability, 4)
            })
            if cumulative_probability >= probability_threshold or p < 1e-8:
                break
            x += 1

        min_spares = x
        total_cost = min_spares * spare_part_cost

        # === Summary Block ===
        st.markdown(f"""
        <div style="padding: 15px; background-color: #eaf6ff; border-radius: 8px; margin-bottom:15px;">
        <p style='color:darkgreen; font-size:22px;'>✅ Minimum Spare Parts for {probability_threshold*100:.1f}% certainty: <strong>{min_spares}</strong></p>
        <p style='color:maroon; font-size:18px;'>💰 Total Cost: <strong>${total_cost:,}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # === Styled Table ===
        st.markdown("### 📊 Probability Table")
        table_html = """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
        }
        th, td {
            text-align: center;
            border: 1px solid #bbb;
            padding: 8px;
        }
        th { background-color: #f7f7f7; color: #333; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
        <table>
        <tr><th>Number of Spares</th><th>Probability</th><th>Cumulative Probability</th></tr>
        """
        for row in probability_table:
            table_html += f"<tr><td>{row['Number of Spares']}</td><td>{row['Probability']:.4f}</td><td>{row['Cumulative Probability']:.4f}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # === Plotting ===
        x_vals = [row["Number of Spares"] for row in probability_table]
        prob_vals = [row["Probability"] for row in probability_table]
        cum_vals = [row["Cumulative Probability"] for row in probability_table]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor("#fafafa")
        ax.plot(x_vals, prob_vals, marker='o', color='royalblue', label='Probability')
        ax.plot(x_vals, cum_vals, marker='o', color='darkorange', label='Cumulative Probability')
        ax.axvline(x=min_spares, color='green', linestyle='--', linewidth=2, label=f'Required Spares = {min_spares}')
        ax.text(min_spares, max(max(prob_vals), max(cum_vals)) * 0.9, f'{min_spares}', color='green', fontsize=12, ha='center', va='bottom')
        ax.set_xlabel("Number of Spares")
        ax.set_ylabel("Probability")
        ax.set_title("Probability & Cumulative Probability vs. Number of Spares", fontsize=16, weight='bold')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # === Inference Texts ===
        st.markdown("### 🔎 Spare Requirement Insights")
        for row in probability_table:
            x = row["Number of Spares"]
            prob = row["Probability"]
            cum_prob = row["Cumulative Probability"]
            risk = 1 - cum_prob
            if prob == 0 and cum_prob == 0: continue

            message = f"For **{x} spare{'s' if x != 1 else ''}**: " \
                    f"**{prob * 100:.2f}% chance** of needing exactly that many, " \
                    f"**{cum_prob * 100:.2f}% confidence** you won’t need more than {x}."
            if risk > 0.2:
                message += f" ⚠️ **{risk * 100:.2f}% risk** of stock shortage."
            st.markdown(f"<div style='margin-bottom:10px; color:#444; font-size:16px;'>{message}</div>", unsafe_allow_html=True)

elif auth_status == False:
    st.error("Username/password is incorrect ❌")

elif auth_status == None:
    st.warning("Please enter your username and password 🔐")
