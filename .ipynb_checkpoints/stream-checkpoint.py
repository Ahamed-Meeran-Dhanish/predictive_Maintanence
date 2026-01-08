# --- Streamlit Real-Time Monitoring with LSTM Model Data and Start/Stop Button ---

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

model = tf.keras.models.load_model('predictive_maintenance_lstm.h5')

st.title("🔍 Real-Time Predictive Maintenance with LSTM Model Data")

TEMP_LIMIT, PRESSURE_LIMIT, FLOW_LIMIT = 70, 150, 60
sequence_length = 30
recent_data = []
time_steps, temp_series, pressure_series, flow_series = [], [], [], []

# Start/Stop button
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button('Start Monitoring'):
    st.session_state.running = True

if st.button('Stop Monitoring'):
    st.session_state.running = False

if st.session_state.running:
    X_data = np.load('X_real_data.npy')  # Assumed to be preprocessed sequences

    for t in range(len(X_data)):
        sample = X_data[t].reshape((1, sequence_length, 3))
        temp, pressure, flow = sample[0, -1, 0], sample[0, -1, 1], sample[0, -1, 2]

        time_steps.append(t)
        temp_series.append(temp)
        pressure_series.append(pressure)
        flow_series.append(flow)

        prediction = model.predict(sample)[0][0]

        if prediction > 0.5:
            st.error(f"⚠️ High Degradation Risk Detected at step {t} ({prediction:.2f})")

        if temp > TEMP_LIMIT:
            st.warning(f"🔥 High Temperature at step {t}: {temp:.2f}°C")
        if pressure > PRESSURE_LIMIT:
            st.warning(f"⚠️ High Pressure at step {t}: {pressure:.2f} Pa")
        if flow > FLOW_LIMIT:
            st.warning(f"💧 High Flow Rate at step {t}: {flow:.2f} L/min")

    row = st.container()

    with row:
        temp_fig = go.Figure()
        temp_fig.add_trace(go.Scatter(x=time_steps, y=temp_series, mode='lines', name='Temperature', line=dict(color='red')))
        temp_fig.update_layout(title="Temperature Over Time", xaxis_title="Time Steps", yaxis_title="°C")
        st.plotly_chart(temp_fig, use_container_width=True)

        pressure_fig = go.Figure()
        pressure_fig.add_trace(go.Scatter(x=time_steps, y=pressure_series, mode='lines', name='Pressure', line=dict(color='blue')))
        pressure_fig.update_layout(title="Pressure Over Time", xaxis_title="Time Steps", yaxis_title="Pa")
        st.plotly_chart(pressure_fig, use_container_width=True)

        flow_fig = go.Figure()
        flow_fig.add_trace(go.Scatter(x=time_steps, y=flow_series, mode='lines', name='Flow Rate', line=dict(color='green')))
        flow_fig.update_layout(title="Flow Rate Over Time", xaxis_title="Time Steps", yaxis_title="L/min")
        st.plotly_chart(flow_fig, use_container_width=True)
else:
    st.info("🛑 Monitoring stopped. Press 'Start Monitoring' to begin.")
