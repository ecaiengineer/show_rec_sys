# Thank you for taking the time to work on the Philo machine learning project.
# We'd like you to create a recommendation model using a dataset we've provided
# that contains actual (anonymized) watch history from a sample of users over a 30 day
# period.

# The following are our requirements:
# - Train a model that given a user id returns a ranked list of show ids.
# - Describe how you evaluated the model and include any data and charts that demonstrate
#   the performance of the model.
# - Identify the steps you took to select the final model, any alternatives you tested,
#   and share and explain any data that will help us understand what choices you made
#   and why at various stages
# - Provide us with instructions on how to train and run the model and use it for inference

# About the dataset:
# - There are 30 consecutive days of data from a sample of 100,000 users
# - A playback session is created every time a user starts video playback in our product.
#   A user may watch multiple assets in a single playback session
# - Asset type is one of CHANNEL, RECORDING, or VOD
#   - If CHANNEL, the user watched a live broadcast
#   - If RECORDING, the user watched a recording of a broadcast
#   - If VOD, the user watched an on-demand asset
# - The ordering of IDs has no meaning

# The project is fairly open ended by design, and we suggest that you take an approach that
# best fits your strengths. It's not required to use all the features in the data set.
# Some example of what success could look like:
# - You provide a Jupyter notebook that includes detailed analysis of some different models, the final model
#   and a explanation of its results and strengths and weakness
# - You create a model that's somewhat simpler than the first example and don't dive as deep into analyis,
#   but you provide code for server that loads the model that we can make requests to for predictions
# The above aren't specifications, but hopefully give you a good idea of how you might approach the project.

# To evaluate your dataset, we'll measure the top 10 recall of model predictions against
# a test dataset containing playback sessions occurring after those in the dataset we have provided you.
# We will measure recall both against the model's ability to predict shows the user has never watched before
# and the ability to predict the next shows watched including those that have been watched before. You may
# choose to optimize for one metric or the other.

# There are many ways you could successfully complete and deliver this project. Try to timebox your effort
# to less than 8 hours. We know there will be more you can and will want to do, and we suggest you
# tell us where you'd go next with the project if you were to work more on it.

# Please don't hesitate to reach out on Slack and ask us questions!
#

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
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
print(df.head(100))
print(df.dtypes)
print(f"Number of unique shows: {df['show_id'].nunique()}")
print(df["show_id"].describe())

all_shows = df['show_id'].unique()
all_asset_types = df['asset_type'].unique()

to_save = {'all_shows': all_shows, 'all_asset_types': all_asset_types}
with open('data/all_shows_and_asset_types.pkl', 'wb') as f:
    pickle.dump(to_save, f)
