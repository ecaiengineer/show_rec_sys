import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import numpy as np

schema = pa.schema([
  ('user_id', pa.int32()),
  ('playback_session_id', pa.int32()),
  ('show_id', pa.int32()),
  ('asset_type', pa.string()),
  ('episode_id', pa.int32()),
  ('day', pa.string()),
  ('time', pa.string()),
  ('watch_minutes', pa.int32()),
])

table = pq.read_table('data/playback_sessions.parquet', schema=schema)
df = table.to_pandas()

# we will focus only on following columns:
# ['user_id', 'show_id', 'asset_type', 'day', 'time', 'watch_minutes']
df = df[['user_id', 'show_id', 'asset_type', 'day', 'time', 'watch_minutes']]

# convert day and time (hh:mm) to a relative time since 0th day
df['relative_day'] = df['day'].astype(float) - 1
df['relative_fraction_of_day'] = df['time'].apply(
    lambda x: float(str(x).split(':')[0])/24.0 + float(str(x).split(':')[1])/24.0/60.0
).astype(float)
df['relative_time'] = df['relative_day'] + df['relative_fraction_of_day']

# use the relative time to sort watching history for each user_id
df = df.sort_values(by=['user_id', 'relative_time'])


# df.groupby('user_id')['show_id'].count().quantile([0.1, 0.2, 0.5])
# 20th percentile is 19 shows watched by a user. Seems like 20 is a reasonable cutoff to use for training the 
# model by using the last 20 shows watched by a user to predict the next show watched.

# creating a dataset where each row is user id and last 20 watched by user, their asset type and watch_minutes 
# and the target is 21st show watched by user. Creating dataset without overlapping segments along the relative time axis.

window_size = 20
dataset = []
inference_dataset = []

for user_id in df['user_id'].unique():
    user_df = df[df['user_id'] == user_id].sort_values(by=['relative_time']).reset_index(drop=True)
    intervals = list(range(len(user_df)-1, 0, -window_size-1))

    for index, i in enumerate(intervals):
        if i - window_size < 0:
            # last segment, which could be incomplete
            segment = user_df.iloc[0:i]
        else:
            segment = user_df.iloc[i-window_size:i]
        
        segment_data = {
            'show_id': segment['show_id'].values,
            'asset_type': segment['asset_type'].values,
            'watch_minutes': segment['watch_minutes'].values,
        }
        target = user_df.iloc[i]['show_id']
        if index == 0:
            # very first segment is reserved for inference
            # remove first element from segment_data and add target to the end
            segment_data['show_id'] = np.concatenate([segment_data['show_id'][1:], [target]])
            segment_data['asset_type'] = np.concatenate([segment_data['asset_type'][1:], [target]])
            segment_data['watch_minutes'] = np.concatenate([segment_data['watch_minutes'][1:], [target]])
            inference_dataset.append({'user_id': user_id, 'inputs': segment_data, 'target': None})
        else:
            dataset.append({'user_id': user_id, 'inputs': segment_data, 'target': target})

# save dataset to pickle
with open(f'data/processed_dataset_window{window_size}.pkl', 'wb') as f:
    pickle.dump(dataset, f)

with open(f'data/inference_dataset_window{window_size}.pkl', 'wb') as f:
    pickle.dump(inference_dataset, f)

print(f"Dataset saved to data/processed_dataset_window{window_size}.pkl")
print(f"Inference dataset saved to data/inference_dataset_window{window_size}.pkl")
print(f"Dataset size: {len(dataset)}")
print(f"Inference dataset size: {len(inference_dataset)}")