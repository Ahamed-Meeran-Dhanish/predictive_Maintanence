# --- Streamlit Real-Time Monitoring with Side-by-Side Live Graphs, Start/Stop, and Reset Button ---

import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import time

model = tf.keras.models.load_model('predictive_maintenance_lstm.h5')

st.title("🔍 Real-Time Predictive Maintenance with Simulated Live Data")

if 'temp_anomalies' not in st.session_state:
    st.session_state.temp_anomalies = 0
if 'pressure_anomalies' not in st.session_state:
    st.session_state.pressure_anomalies = 0
if 'flow_anomalies' not in st.session_state:
    st.session_state.flow_anomalies = 0

st.markdown(f"### 🔢 Total Anomalies Detected")
st.markdown(f"- 🔥 Temperature: {st.session_state.temp_anomalies}")
st.markdown(f"- ⚠️ Pressure: {st.session_state.pressure_anomalies}")
st.markdown(f"- 💧 Flow Rate: {st.session_state.flow_anomalies}")

TEMP_LIMIT, PRESSURE_LIMIT, FLOW_LIMIT = 70, 150, 60
sequence_length = 30

if 'running' not in st.session_state:
    st.session_state.running = False
if 'time_steps' not in st.session_state:
    st.session_state.time_steps = []
if 'temp_series' not in st.session_state:
    st.session_state.temp_series = []
if 'pressure_series' not in st.session_state:
    st.session_state.pressure_series = []
if 'flow_series' not in st.session_state:
    st.session_state.flow_series = []
st.session_state.temp_anomalies = 0
st.session_state.pressure_anomalies = 0
st.session_state.flow_anomalies = 0
if 'temp_anomalies' not in st.session_state:
    st.session_state.temp_anomalies = 0
if 'pressure_anomalies' not in st.session_state:
    st.session_state.pressure_anomalies = 0
if 'flow_anomalies' not in st.session_state:
    st.session_state.flow_anomalies = 0

if st.button('Start Monitoring'):
    st.session_state.running = True

if st.button('Stop Monitoring'):
    st.session_state.running = False

if st.button('Reset Data'):
    st.session_state.time_steps = []
    st.session_state.temp_series = []
    st.session_state.pressure_series = []
    st.session_state.flow_series = []
    st.session_state.temp_anomalies = 0
    st.session_state.pressure_anomalies = 0
    st.session_state.flow_anomalies = 0
    st.session_state.flow_series = []

col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    temp_alert = st.empty()
    temp_chart = st.empty()
with col2:
    pressure_alert = st.empty()
    pressure_chart = st.empty()
with col3:
    flow_alert = st.empty()
    flow_chart = st.empty()

if st.session_state.running:
    recent_data = []
    for t in range(1, 501):
        temp = 50 + np.random.normal(0, 5) + (20 if t % 100 == 0 else 0)
        pressure = 100 + np.random.normal(0, 10) + (30 if t % 150 == 0 else 0)
        flow = 30 + np.random.normal(0, 3) + (15 if t % 200 == 0 else 0)

        recent_data.append([temp, pressure, flow])
        if len(recent_data) > sequence_length:
            recent_data.pop(0)

        st.session_state.time_steps.append(t)
        st.session_state.temp_series.append(temp)
        st.session_state.pressure_series.append(pressure)
        st.session_state.flow_series.append(flow)

        if len(recent_data) == sequence_length:
            sample = np.array(recent_data).reshape((1, sequence_length, 3))
            prediction = model.predict(sample)[0][0]

            if prediction > 0.5:
                st.error(f"⚠️ High Degradation Risk Detected at step {t} ({prediction:.2f})")

        if temp > TEMP_LIMIT:
            st.session_state.temp_anomalies += 1
            temp_alert.warning(f"🔥 High Temperature at step {t}: {temp:.2f}°C (Total: {st.session_state.temp_anomalies})")
        else:
            temp_alert.empty()
        if pressure > PRESSURE_LIMIT:
            st.session_state.pressure_anomalies += 1
            pressure_alert.warning(f"⚠️ High Pressure at step {t}: {pressure:.2f} Pa (Total: {st.session_state.pressure_anomalies})")
        else:
            pressure_alert.empty()
        if flow > FLOW_LIMIT:
            st.session_state.flow_anomalies += 1
            flow_alert.warning(f"💧 High Flow Rate at step {t}: {flow:.2f} L/min (Total: {st.session_state.flow_anomalies})")
        else:
            flow_alert.empty()

        temp_fig = go.Figure()
        temp_fig.add_trace(go.Scatter(x=st.session_state.time_steps, y=st.session_state.temp_series, mode='lines', name='Temperature', line=dict(color='red')))
        if temp > TEMP_LIMIT:
            temp_fig.add_trace(go.Scatter(x=[t], y=[temp], mode='markers', name='Anomaly', marker=dict(color='purple', size=15)))
        temp_fig.update_layout(title="Temperature Over Time", xaxis_title="Time Steps", yaxis_title="°C")
        temp_chart.plotly_chart(temp_fig, use_container_width=False, height=400)

        pressure_fig = go.Figure()
        pressure_fig.add_trace(go.Scatter(x=st.session_state.time_steps, y=st.session_state.pressure_series, mode='lines', name='Pressure', line=dict(color='blue')))
        if pressure > PRESSURE_LIMIT:
            pressure_fig.add_trace(go.Scatter(x=[t], y=[pressure], mode='markers', name='Anomaly', marker=dict(color='orange', size=10)))
        pressure_fig.update_layout(title="Pressure Over Time", xaxis_title="Time Steps", yaxis_title="Pa")
        pressure_chart.plotly_chart(pressure_fig, use_container_width=False, height=400)

        flow_fig = go.Figure()
        flow_fig.add_trace(go.Scatter(x=st.session_state.time_steps, y=st.session_state.flow_series, mode='lines', name='Flow Rate', line=dict(color='green')))
        if flow > FLOW_LIMIT:
            flow_fig.add_trace(go.Scatter(x=[t], y=[flow], mode='markers', name='Anomaly', marker=dict(color='orange', size=10)))
        flow_fig.update_layout(title="Flow Rate Over Time", xaxis_title="Time Steps", yaxis_title="L/min")
        flow_chart.plotly_chart(flow_fig, use_container_width=False, height=400)

        time.sleep(0.5)
else:
    st.info("🛑 Monitoring stopped. Press 'Start Monitoring' to begin.")
