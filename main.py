import os
import pandas as pd
import numpy as np
from src.ingestion.ims_loader import load_raw_file, parse_filename_timestamp
from src.features.signal_math import calculate_time_domain_features
from src.models.anomaly_algo import BearingAnomalyDetector

try:
    from src.utils.plotting import plot_sensor_history, plot_anomaly_score
except ImportError:
    pass

# config
DATA_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed/merged_bearing_data.csv'
MODEL_PATH = 'saved_models/iso_forest.pkl'

def main():
    print("--- industrial ai: bearing fault detection pipeline ---")
    
    # 1. load and process raw data
    print(f"[step 1] reading files from {DATA_PATH}...")
    
    # get all files in data/raw, sort them by time
    files = sorted([f for f in os.listdir(DATA_PATH) if not f.startswith('.')])
    
    if not files:
        print("error: no files found in data/raw! did you put them there?")
        return

    feature_list = []
    
    # loop through files
    print(f"processing {len(files)} files. please wait...")
    
    for i, filename in enumerate(files):
        # simple progress bar
        if i % 100 == 0:
            print(f"  -> processing file {i}/{len(files)}")
            
        filepath = os.path.join(DATA_PATH, filename)
        timestamp = parse_filename_timestamp(filename)
        
        if timestamp is None:
            continue
            
        # load raw vibration signal
        signal = load_raw_file(filepath)
        
        if signal is not None:
            # calculate math features (rms, kurtosis, etc)
            feats = calculate_time_domain_features(signal)
            feats['timestamp'] = timestamp
            feature_list.append(feats)
    
    # create dataframe
    df = pd.DataFrame(feature_list)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"data processing complete. shape: {df.shape}")
    
    # save the clean data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH)
    print(f"saved processed data to {PROCESSED_PATH}")

    # 2. train model (anomaly detection)
    print("[step 2] training isolation forest...")
    
    # we assume the first 20% of data is "healthy" (calibration period)
    train_size = int(len(df) * 0.2)
    train_data = df.iloc[:train_size]
    
    # features to use for training
    feature_cols = ['rms', 'kurtosis', 'skewness', 'peak_to_peak', 'crest_factor']
    
    detector = BearingAnomalyDetector(contamination=0.05)
    detector.train(train_data[feature_cols])
    
    # 3. predict on the whole dataset
    print("[step 3] running inference on full lifecycle...")
    df['anomaly_score'] = detector.get_anomaly_score(df[feature_cols])
    df['prediction'] = detector.predict(df[feature_cols]) # -1 is anomaly
    
    # 4. save model
    detector.save_model(MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")
    
    # 5. generate plots
    print("[step 4] generating plots...")
    os.makedirs('plots', exist_ok=True)
    
    try:
        # we only plot the first bearing's RMS for clarity in the history plot
        # but the model uses the average stats we calculated
        plot_sensor_history(df[['rms']], save_path='plots/sensor_history.png')
        plot_anomaly_score(df['anomaly_score'], -0.5, save_path='plots/anomaly_detection.png')
        print("plots saved to plots/ folder.")
    except NameError:
        print("plotting skipped (module not imported).")
    except Exception as e:
        print(f"plotting failed: {e}")

    print("\n--- pipeline success ---")
    print("run 'uvicorn deploy.app:app --reload' to start the api.")

if __name__ == "__main__":
    main()
