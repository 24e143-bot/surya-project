# SURYA | IoT Solar PV Intelligent Reconfiguration System

**SURYA** is a Digital Twin dashboard designed for real-time monitoring and PSO-based reconfiguration of Solar PV arrays. It addresses the problem of power loss due to partial shading and faults using AI (LSTM) and Optimization (PSO).

## 🚀 Key Features
- **Fault Injection Control:** Simulate real-world shading and hotspots.
- **Intelligent Reconfiguration:** PSO-driven TCT (Total-Cross-Tied) layout optimization.
- **PI-LSTM Integration:** Predictive monitoring of system health and power output.
- **Interactive Heatmap:** Visual irradiance distribution before and after optimization.
- **Automated KPI Tracking:** Instant calculation of Power Recovery and Efficiency Gains.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly (Dynamic Graphs & Heatmaps)
- **Algorithm:** Particle Swarm Optimization (PSO)
- **Predictive Model:** PI-LSTM (AI)
- **Data Handling:** NumPy, Pandas

## 📦 Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/surya-digital-twin.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run surya_app.py
   ```

