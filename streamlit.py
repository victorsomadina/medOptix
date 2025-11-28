import streamlit as st
import requests
from datetime import date
import pandas as pd

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="MedOptix Forecast", page_icon="🩺")

st.title("MedOptix Forecast UI")
st.write("Enter patient / hospital details to get Admissions Forecast.")

with st.form("prediction_form"):
    st.subheader("Context")

    hospital = st.selectbox(
        "Hospital",
        options=[
            "Helsinki Central Hospital",
            "Tampere City Hospital",
            "Turku University Hospital",
            "Oulu Regional Hospital",
            "Kuopio Medical Center",
        ],
        index=0,
    )

    ward_code = st.selectbox(
        "Ward code",
        options=["ED", "ICU", "MED", "SURG"],
        index=0,
    )

    st.subheader("Input features")

    age = st.number_input("Age", value=54.89)
    previous_day_occupancy = st.number_input("Previous day occupancy", value=34.0)
    previous_day_overflow = st.number_input("Previous day overflow", value=42.0)
    previous_day_avg_wait = st.number_input("Previous day average wait (mins)", value=227.0)

    arrival_source = st.selectbox(
        "Arrival source",
        options=["self", "referral", "ambulance"],
        index=0
    )

    outcome = st.selectbox(
        "Outcome",
        options=["discharged", "admitted", "transferred"],
        index=0
    )

    sex = st.selectbox("Sex", options=["M", "F"], index=0)

    base_beds = st.number_input("Base beds", value=30, step=1)
    effective_capacity = st.number_input("Effective capacity", value=34, step=1)
    staffing_index = st.number_input("Staffing index", value=0.927)
    previous_day_discharges = st.number_input("Previous day discharges", value=33.0)
    previous_day_admission_rate_per_bed = st.number_input(
        "Previous day admission rate per bed",
        value=2.233
    )

    steps = st.number_input(
        "Forecast horizon (days ahead)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
    )

    start_date = st.date_input(
        "Forecast start date",
        value=date.today()
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "hospital": hospital,   
        "ward_code": ward_code,  

        "age": age,
        "previous_day_occupancy": previous_day_occupancy,
        "previous_day_overflow": previous_day_overflow,
        "previous_day_avg_wait": previous_day_avg_wait,
        "arrival_source": arrival_source,
        "outcome": outcome,
        "sex": sex,
        "base_beds": base_beds,
        "effective_capacity": effective_capacity,
        "staffing_index": staffing_index,
        "previous_day_discharges": previous_day_discharges,
        "previous_day_admission_rate_per_bed": previous_day_admission_rate_per_bed,
        "steps": int(steps),
        "start_date": start_date.isoformat(),
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction successful")

            forecast_values = result.get("forecast", [])

            st.write(f"**Hospital:** {hospital} | **Ward code:** {ward_code}")
            st.write("**Forecast values:**", forecast_values)

            if isinstance(forecast_values, list) and len(forecast_values) > 0:
                dates = pd.date_range(
                    start=start_date,
                    periods=len(forecast_values),
                    freq="D",
                )

                df = pd.DataFrame(
                    {"Admissions forecast": forecast_values},
                    index=dates
                )

                st.line_chart(df)

        else:
            st.error(f"API error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Request failed: {e}")
