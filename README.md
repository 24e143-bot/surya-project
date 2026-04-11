#  SURYA | Sustainable Unified Renewable Yield & Adaptive Grid Architecture

**SURYA** is an industry-grade Digital Twin dashboard designed for real-time monitoring, maintenance prediction, and adaptive grid balancing. It bridges the gap between **Solar PV Efficiency** and **Distribution Feeder Stability** using a multi-layer optimization framework.

---

##  Project Abstract
The modern energy transition faces a dual crisis: localized power loss in solar arrays due to shading/faults and systemic grid instability caused by uneven power injection. **SURYA** addresses these by implementing a **"Local-First Correction"** paradigm. By solving PV mismatches at the source using **PSO-driven reconfiguration**, it prevents phase imbalances from ever reaching the utility feeder, transforming solar energy from a disruptive source into a self-healing grid participant.

---

##  Key Features

### 🔹 Layer 1: Solar PV Intelligence (Self-Healing)
- **Fault Injection Control:** Simulate real-world shading, soiling, and hotspot faults to test system resilience.
- **Intelligent PSO Reconfiguration:** Uses **Particle Swarm Optimization** to re-map a Total-Cross-Tied (TCT) switching matrix, recovering up to 30% of lost power.
- **Physics-Informed RUL:** Predicts **Remaining Useful Life** and health indices based on real-time thermal and electrical stress models.
- **Automated Maintenance Alerts:** Logic-driven recommendations (e.g., Priority Inspection) based on predicted asset failure horizons.

### 🔹 Layer 2: Adaptive Grid Stability (Feeder Optimization)
- **Feeder Digital Twin:** Real-time simulation of an **IEEE 13-bus distribution system** with phase-level granularity.
- **Hosting Capacity Support:** Dynamic monitoring of how much renewable energy the grid can safely absorb.
- **Adaptive Hysteresis Switching:** Intelligent logic that toggles between feeders to maintain stability during extreme solar fluctuations.
- **Phase Balance Monitoring:** Real-time visualization and metric tracking of **Phase R, Y, and B** equilibrium.

---

##  Tech Stack
- **Frontend:** Streamlit (Premium Dark-Theme Dashboard)
- **Visualization:** Plotly (Dynamic trends, Heatmaps, and Phase bar charts)
- **Algorithms:** Particle Swarm Optimization (PSO), Hysteresis Control Logic
- **Predictive Model:** PI-LSTM (AI-based generation prediction)
- **Backend:** Python, NumPy, Pandas (High-fidelity physical simulation)
- **Communication (Architecture):** Designed for MQTT-based IoT data flow

---

##  Impact & Business Value
- **OpEx Reduction:** Reduces high-wear mechanical switching operations on the grid by solving faults locally.
- **Revenue Protection:** Prevents asset destruction (hotspots) and maximizes yield in shaded environments.
- **Grid Resilience:** Increases the "Hosting Capacity" of existing neighborhoods, allowing for more solar installations without infrastructure upgrades.

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/surya-project.git
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Digital Twin dashboard:**
   ```bash
   streamlit run surya_app.py
   ```

---

## The SURYA Workflow
**Solar Fault Detected** ➔ **Local PSO Reconfiguration** ➔ **PV Power Restored** ➔ **Feeder Stability Evaluated** ➔ **Grid Balanced**

---

### **Project Theme:** IoT in Sustainability | Smart Grid Infrastructure
---
