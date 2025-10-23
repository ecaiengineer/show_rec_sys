# TV Show Recommendation System

A deep learning-based recommendation system that predicts the next TV show a user is likely to watch based on their viewing history. The system uses an LSTM neural network to model sequential viewing patterns.

## Overview

This project implements a sequential recommendation model that:
- Takes a user's watch history (last 20 shows watched) as input (design rationale: [here](#key-design-decisions))
- Uses LSTM architecture to capture viewing patterns
- Predicts the next show the user is likely to watch
- Optimizes for **Top-10 Recall** - whether the actual next show appears in the top 10 predictions


## Data Processing

### Source Data 
`playback_sessions.parquet` - 100,000 users over 30 days

### Feature Sequence Creation Process
1. Sort viewing history chronologically by converting day/time to relative time
2. Create non-overlapping segments of 20 consecutive shows per user (step size = 21)
3. Each sample: 20 input shows → predict 21st show
4. Most recent segment reserved for inference; remaining for train/val/test
5. Data split: Train (70%), Validation (10%), Test (20%) - random splits


## Model

### Model Architecture
```
Input Sequence → [Show Embeddings + Asset Type Embeddings + Watch Minutes] → LSTM (x N) → Linear Layer → Predictions
```
### Input Features
1. **Show ID Sequence**: Last 20 shows watched by the user
2. **Asset Type Sequence**: Last 20 Asset Type of each viewing 
3. **Watch Duration**: Last 20; of Minutes watched for each show

### Architecture Components
- **Embedding Layers**: Embedding layers used to convert sparse categorical IDs into dense, low-dimensional vectors. The embeddings learn meaningful relationships between shows during training and they are optimized for the prediction task.
- **LSTM Layer**: To model the sequential relationships in the data
- **Output Layer**: Linear projection to predict next show
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam 

### Model Hyperparameters

| Hyperparameter | Description |
|----------------|-------------|
| Show Embedding Dimension |  Dimensionality of show ID embeddings |
| Asset Type Embedding Dimension | Dimensionality of asset type embeddings |
| LSTM Hidden Dimension |  Number of hidden units in LSTM layer |
| Dropout Rate | Dropout probability for regularization |
| Number of LSTM Layers |  Number of LSTM layers |

### Training Configuration


| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Epochs | 10 | Number of training epochs |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Window Size | 20 | Number of previous shows in input sequence |
| Batch Size | 512 | Number of samples per training batch |


## Experiments & Results

| Model | LSTM Layers | Embedding Dim (both) | Hidden Dim | Dropout | Train Top-10 Recall | Val Top-10 Recall | Test Top-10 Recall |
|-------|-------------|---------------|------------|---------|---------------------|-------------------|---------------------|
| lstm_1     | 1           | 32            | 128        | 0.0     |   74.9%               | 75.4%             | 75.2%               |
| lstm_2     | 2           | 64            | 128        | 0.2     |   77.9%               | **77.7%**            | 77.4%           |
| lstm_3 | 3 | 128 | 256 | 0.3 |77.1% | 77.1% | 76.8%|

lstm_2 was chosen as the final model due to its best peformance.


## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the Data
Run the preprocessing script to create training dataset:
```bash
python create_dataset.py
```

This creates:
- `data/processed_dataset_window20.pkl`: Processed sequences stored before training
- `data/inference_dataset_window20.pkl`: Inference sequences to be used by inference function
- `data/all_shows_and_asset_types.pkl`: Show and asset type mappings

### 2. Train the Model
```bash
python main.py
```
### 3. Running Inference

Use the `LSTMRecommender` class to make predictions for users:

#### Python/Jupyter Notebook
```python
from inference import LSTMRecommender

# Initialize the recommender with trained model
recommender = LSTMRecommender(
    check_point_file='artifacts/lstm_2.pth'
)

# Get ranked list of shows for a user
user_id = 8319
ranked_shows = recommender.predict_shows(user_id=user_id) # gives ranked list for all shows

# Display top 10 recommendations
print(f"Top 10 shows for user {user_id}: {ranked_shows[:10]}")
```

Output:
```
first 10 ranked shows for user 8319: [48, 196, 698, 690, 876, 222, 1157, 49, 268, 1376]
```

#### Command Line
```bash
python inference.py
```

The `predict_shows()` method returns a ranked list of all shows, ordered by predicted relevance from most to least likely.



## Key Design Decisions

#### Window Size = 20
- Arrived at by looking at the histogram of number of shows watched in dataset and choosing 20th percentile (which is ~20 shows)
- Balances context and computational efficiency

#### LSTM Architecture
- Models sequential temporal dependencies
- Captures both short-term and long-term viewing patterns


## Future Improvements

1. **Model Enhancements**:
   - Deeper LSTM (multiple layers) to see if model performance can be improved.
   - Attention mechanism/transformer architecture could be experimented with. 
   - User embeddings for personalization could be used for more personalization. Currently this model treats each watch history as an independent observation.

2. **Training Improvements**:
   - A systematic grid search (or bayesian optimiation) could be performed to find the optimal hyperparameters
   - Learning rate scheduling, and early stopping methods could be tried.

3. **Feature Engineering**:
   - Episode information (binge-watching patterns), Show genre/category and show popularity metrics could be incorporated.
   - Intuitively it makes sense that viewing history importance decays with time, so considering some measure of decay in the input could be interesting to try.

4. **Cold-start Problem**:
   - Since this model expects user watching history for prediction, it suffers from cold start problem for new user, that can be remediated by considering their preferences metadata and recommending shows based on what similar users may have liked.

## License

See LICENSE file for details.
