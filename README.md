## 🔧 Predictive Maintenance System using LSTM

ABB Hackathon Project | Team Ideators

## 📌 Overview

This project is a Predictive Maintenance System developed during the ABB Hackathon by Team Ideators.
The solution uses a Long Short-Term Memory (LSTM) deep learning model to predict equipment degradation risk based on time-series sensor data.

The project focuses on early fault detection, anomaly monitoring, and real-time visualization, helping industries reduce unplanned downtime and maintenance costs.

## 🏆 Outcome:
This project earned our team an internship opportunity at ABB.

## 🚀 Key Features

📈 Time-Series Prediction using LSTM

🔍 Real-time sensor monitoring

⚠️ Anomaly detection for:

   Temperature

   Pressure

   Flow rate

📊 Live interactive graphs using Streamlit & Plotly

⏯️ Start / Stop / Reset controls

🧠 Pre-trained LSTM model for degradation risk prediction

## 🧠 Model Details

Model Type: LSTM (Long Short-Term Memory)

Framework: TensorFlow / Keras

Input: Sequential sensor data
(Temperature, Pressure, Flow Rate)

Sequence Length: 30 time steps

Output: Degradation risk score (probability)

The trained model is stored as:

predictive_maintenance_lstm.h5


## 📊 Sensor Data

Sensor data is simulated using NumPy

Mimics real industrial signals:

Temperature (°C)

Pressure (Pa)

Flow Rate (L/min)

Sudden spikes are intentionally introduced to test anomaly detection

## 🖥️ Real-Time Monitoring Dashboard

The Streamlit application provides:

Live updating graphs for all sensors

Anomaly counters for each sensor

Alerts for:

High temperature

High pressure

High flow rate

Degradation risk alerts from the LSTM model

## ▶️ How to Run the Project

1️⃣ Install Dependencies
pip install streamlit tensorflow numpy plotly

2️⃣ Run the Streamlit App
streamlit run stream.py

3️⃣ Interact with the Dashboard

Click Start Monitoring

Observe live sensor graphs

View anomaly alerts and degradation predictions

Use Stop or Reset as needed

## 🎯 Use Case

Industrial equipment health monitoring

Predictive maintenance systems

Smart asset management

Industry 4.0 applications

## 👥 Team

Team Name: Ideators
Event: ABB Hackathon

## 🏁 Conclusion

This project demonstrates how deep learning and real-time monitoring can be combined to build an effective predictive maintenance solution.
It reflects practical industry use cases and aligns closely with ABB’s smart asset management vision.

This project demonstrates how deep learning and real-time monitoring can be combined to build an effective predictive maintenance solution.
It reflects practical industry use cases and aligns closely with ABB’s smart asset management vision.
