import numpy as np
from scipy.stats import skew, kurtosis

def calculate_time_domain_features(signal_data):
    """
    computes statistical features for a given vibration signal.
    
    args:
        signal_data (np.array): raw vibration signal from the sensor.
        
    returns:
        dict: feature vector containing rms, kurtosis, etc.
    """
    # rms (root mean square) - indicates general energy level
    rms = np.sqrt(np.mean(signal_data**2))
    
    # kurtosis - key for early fault detection (impulsive shocks)
    kurt = kurtosis(signal_data, fisher=False)
    
    # skewness - measures asymmetry of the signal distribution
    skw = skew(signal_data)
    
    # peak-to-peak - absolute range of vibration
    p2p = np.max(signal_data) - np.min(signal_data)
    
    # crest factor - ratio of peak to rms (impact severity)
    crest_factor = np.max(np.abs(signal_data)) / rms if rms > 0 else 0
    
    return {
        'rms': rms,
        'kurtosis': kurt,
        'skewness': skw,
        'peak_to_peak': p2p,
        'crest_factor': crest_factor
    }
