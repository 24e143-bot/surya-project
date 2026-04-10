import numpy as np
import pandas as pd

def get_feeder_data():
    """
    Simulates and returns feeder-side digital twin data for the SURYA dashboard.
    Models hosting capacity, voltage safety, feeder switching, and IoT metrics.
    """
    n_points = 50  # Generating 50 data points for the trend
    
    # 1. Hosting capacity (Range: 70 kW to 150 kW)
    t = np.linspace(0, 10, n_points)
    base_hc = 110 + 30 * np.sin(t)
    noise = np.random.normal(0, 8, n_points)
    
    hc_raw = np.clip(base_hc + noise, 70, 150)
    
    # Smooth variation using pandas rolling average
    hc_smooth_series = pd.Series(hc_raw).rolling(window=5, min_periods=1).mean()
    hc_smooth = np.clip(hc_smooth_series.values, 70, 150)
    
    # 2. Feeder switching logic (Hysteresis based on Hosting Capacity)
    # HC > 120 -> feeder 2
    # HC < 90 -> feeder 1
    # Else retain previous feeder
    feeder_array = np.zeros(n_points, dtype=int)
    current_feeder = 1
    
    for i in range(n_points):
        if hc_smooth[i] > 120:
            current_feeder = 2
        elif hc_smooth[i] < 90:
            current_feeder = 1
        feeder_array[i] = current_feeder
        
    # 3. Voltage safety limits
    vmax = round(np.random.uniform(1.00, 1.08), 3)
    vmin = round(np.random.uniform(0.92, 1.00), 3)
    
    if vmax > 1.05 or vmin < 0.95:
        status = "VIOLATION"
    else:
        status = "SAFE"
        
    # 4. Inverter power (Range: 80 kW to 140 kW)
    inverter_kw = round(np.random.uniform(80, 140), 2)
    
    # 5. Phase visualization (Conceptual values near 1.0 pu)
    # Tightened range for physical realism: 0.97 to 1.03 results in ~0.01-0.03 imbalance
    phase_r = round(np.random.uniform(0.98, 1.02), 3)
    phase_y = round(np.random.uniform(0.97, 1.03), 3)
    phase_b = round(np.random.uniform(0.98, 1.02), 3)
    
    # 6. Grid relief metric (Range: 5 to 20)
    switch_saved = int(np.random.uniform(8, 18))
    total_ops = switch_saved + int(np.random.uniform(2, 5))
    
    # 7. IoT layer metrics
    latency_ms = int(np.random.uniform(100, 200))
    sensor_status = np.random.choice(["ONLINE", "OFFLINE"], p=[0.95, 0.05])
    communication_status = np.random.choice(["Stable", "Delayed"], p=[0.85, 0.15])
    
    return {
        "hc_raw": hc_raw.tolist(),
        "hc_smooth": hc_smooth.tolist(),
        "feeder_array": feeder_array.tolist(),
        "vmax": vmax,
        "vmin": vmin,
        "status": status,
        "inverter_kw": inverter_kw,
        "phase_r": phase_r,
        "phase_y": phase_y,
        "phase_b": phase_b,
        "switch_saved": switch_saved,
        "total_ops": total_ops,
        "latency_ms": latency_ms,
        "sensor_status": sensor_status,
        "communication_status": communication_status
    }
