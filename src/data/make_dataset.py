import pandas as pd
from typing import Tuple
from glob import glob


def read_data_from_files(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Extract gyroscope and accellerometer data from .CSV files contained in path to two dataframes.
    epoch (ms) is use to create a datetime index. participant, category, and label are extracted from file names.
    '''
    files = glob(path + '*.csv')
    
    print('Number of CSV files identified:', len(files))

    # Read all files
    acc_df = pd.DataFrame() 
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split('-')[0].replace(path, '')
        category = f.split('-')[2].split('_')[0].rstrip('123')
        label = f.split('-')[1]
        df = pd.read_csv(f)
        df['participant'] = participant
        df['category'] = category
        df['label'] = label
        
        if 'Accelerometer' in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        
        if 'Gyroscope' in f:   
            df['set'] = gyr_set
            gyr_set += 1 
            gyr_df = pd.concat([gyr_df, df])

    # Working with datetimes
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del gyr_df['epoch (ms)']
    del gyr_df['time (01:00)']
    del gyr_df['elapsed (s)']
    
    return acc_df, gyr_df

path = '../../data/raw/MetaMotion/'

acc_df, gyr_df = read_data_from_files(path)


# Merging datasets
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

print('Shape acc_df:', acc_df.shape)
print('Shape gyr_df:', gyr_df.shape)
print('Shape data_merged:', data_merged.shape)
print('Shape data_merged with simple dropna:', data_merged.dropna().shape)

data_merged.columns = [
    'acc_x',
    'acc_y', 
    'acc_z', 
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participant',
    'category',
    'label',
    'set'
]

data_merged.info()

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
# Even though gyr is exactly 2x Hz of acc, very few datetimes are identical.
# Want to keep as much resolution as possible and have complete data for every row 

sampling = {
    'acc_x': 'mean',
    'acc_y': 'mean',
    'acc_z': 'mean',
    'gyr_x': 'mean',
    'gyr_y': 'mean',
    'gyr_z': 'mean',
    'participant': 'last',
    'category': 'last',
    'label': 'last',
    'set': 'last'
}

# Test on subset:
# data_merged[:100].resample(rule='200ms').apply(sampling)

# Trick to avoid giant df with tons of NaNs:
days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
data_resampled = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

data_resampled['set'] = data_resampled['set'].astype('int')

# Export dataset parquet file. DE video recommeded pickle file but had trouble..
# pylance would not reocognize type and suggest autocompletion
# Some possible for mischief with binary-type data files
# So, using parquet...
export_path = '../../data/interim/01_data_processed.parquet'
data_resampled.to_parquet(export_path)