import pandas as pd
import os
from datetime import datetime

def parse_filename_timestamp(filename):
    """
    nasa ims filenames are formatted as YYYY.MM.DD.HH.MM.SS
    we need to convert this to a datetime object.
    """
    try:
        # strip format if present
        clean_name = filename.replace('.txt', '')
        return datetime.strptime(clean_name, '%Y.%m.%d.%H.%M.%S')
    except ValueError:
        return None

def load_raw_file(filepath):
    """
    loads a single bearing data file. 
    the nasa dataset has no headers and is tab-separated.
    """
    try:
        # reading only the first bearing channel (col 0) for simplicity
        df = pd.read_csv(filepath, sep='\t', header=None, usecols=[0])
        return df.values.flatten()
    except Exception as e:
        print(f"error reading {filepath}: {e}")
        return None
