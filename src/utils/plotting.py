import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# set the style globally to look like a scientific paper
plt.style.use('ggplot')

def plot_sensor_history(df: pd.DataFrame, save_path=None):
    """
    plots the history of the rms features.
    this shows the degradation of the machine over time.
    """
    plt.figure(figsize=(12, 6))
    
    # plot rms for each bearing (assuming columns 0-3 are the bearings)
    # usually bearing 1 is the one that fails in test 2
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, alpha=0.7)
        
    plt.title('Bearing Vibration History (RMS)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Vibration Amplitude (RMS)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"saved sensor history plot to {save_path}")
    
    # close to free up memory
    plt.close()

def plot_anomaly_score(scores, threshold, save_path=None):
    """
    plots the anomaly score output by the isolation forest.
    shows exactly when the model flagged the failure.
    """
    plt.figure(figsize=(12, 6))
    
    # create a time axis (just an index for now)
    x_axis = np.arange(len(scores))
    
    plt.plot(x_axis, scores, label='Anomaly Score', color='blue', alpha=0.6)
    
    # plot the threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label='Decision Boundary')
    
    # fill the area where failure is detected
    # scores below threshold are anomalies in isolation forest
    anomaly_indices = np.where(scores < threshold)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, scores[anomaly_indices], color='red', s=10, label='Detected Anomaly')

    plt.title('AI Health Assessment: Anomaly Score Evolution', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Isolation Forest Score (Lower is Worse)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"saved anomaly score plot to {save_path}")
    
    plt.close()