# TV Show Recommendation System

A deep learning-based recommendation system that predicts the next TV show a user is likely to watch based on their viewing history. The system uses an LSTM neural network to model sequential viewing patterns.

## Project Overview

This project implements a sequential recommendation model that:
- Takes a user's watch history (last 20 shows watched) as input
- Uses LSTM architecture to capture viewing patterns
- Predicts the next show the user is likely to watch
- Optimizes for **Top-10 Recall** - whether the actual next show appears in the top 10 predictions

## What Does `main.py` Do?

The `main.py` script is the entry point for training the recommendation model. Here's a step-by-step breakdown:

### 1. Configuration Setup (Lines 6-12)
```python
dataset_path = 'data/processed_dataset_window20.pkl'
all_shows_and_asset_types_path = 'data/all_shows_and_asset_types.pkl'
batch_size = 512
num_epochs = 5
learning_rate = 0.001
window_size = 20
```
Sets up hyperparameters for training:
- **Window size**: Number of previous shows to consider (20)
- **Batch size**: Number of samples per training batch (512)
- **Learning rate**: Step size for gradient descent (0.001)
- **Epochs**: Number of complete passes through the dataset (5)

### 2. Load Show and Asset Type Mappings (Lines 14-18)
```python
with open(all_shows_and_asset_types_path, 'rb') as f:
    all_shows_and_asset_types = pickle.load(f)

all_shows = all_shows_and_asset_types['all_shows']
all_asset_types = all_shows_and_asset_types['all_asset_types']
```
Loads the complete lists of unique shows and asset types (CHANNEL, RECORDING, VOD) from the dataset.

### 3. Device Selection (Lines 20-22)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```
Automatically detects and uses GPU if available, otherwise uses CPU.

### 4. Create Data Loaders (Lines 24-32)
Calls `create_data_loaders()` from `data_loader.py` to:
- Load the preprocessed dataset
- Split data into train (70%), validation (10%), and test (20%) sets
- Create PyTorch DataLoader objects for efficient batch processing
- Handle padding/truncation of sequences to window_size

### 5. Initialize the Model (Lines 34-45)
Creates a `RecSysModel` instance with:
- **Show embedding**: 32-dimensional learned representation for each show
- **Asset type embedding**: 32-dimensional representation for each asset type
- **LSTM hidden dimension**: 128 units
- **Output dimension**: Number of all shows + 1 (for padding)

The model architecture:
```
Input Sequence → [Show Embeddings + Asset Type Embeddings + Watch Minutes] → LSTM → Linear Layer → Predictions
```

### 6. Train the Model (Lines 49-58)
Calls `train_model()` from `model.py` to:
- Train the model for the specified number of epochs
- Optimize using Adam optimizer with CrossEntropy loss
- Track training and validation metrics (loss and top-10 recall)
- Print progress after every 100 batches

### 7. Evaluation (Lines 60-65, currently commented out)
Optional evaluation on the test set using `evaluate_model()` to measure final performance.

## Functions Called by `main.py`

### From `data_loader.py`

#### `create_data_loaders(dataset_path, all_shows, all_asset_types, batch_size, window_size)`
**Purpose**: Creates train, validation, and test data loaders.

**Process**:
1. Loads the preprocessed dataset from pickle file
2. Calls `split_dataset()` to divide data into train/val/test sets
3. Creates three `RecSysDataset` instances (train, val, test)
4. Wraps datasets in PyTorch `DataLoader` objects for batching

**Returns**: `(train_loader, val_loader, test_loader)`

#### `split_dataset(dataset, test_size=0.2, val_size=0.1, random_state=42)`
**Purpose**: Splits the dataset into training, validation, and test sets.

**Process**:
1. First split: Separates test set (20%) from train+val (80%)
2. Second split: From remaining 80%, creates train (70% total) and val (10% total)
3. Uses sklearn's `train_test_split` with fixed random seed for reproducibility

**Returns**: `(train_data, val_data, test_data)`

#### `RecSysDataset` (Class)
**Purpose**: PyTorch Dataset that handles data preprocessing and batching.

**Key Methods**:
- `__init__()`: Initializes show-to-index and asset-type-to-index mappings
- `__getitem__(idx)`: Returns a single preprocessed sample with:
  - Show ID sequence (padded/truncated to window_size)
  - Asset type sequence
  - Watch duration sequence
  - Target (next show to predict)
- `__len__()`: Returns total number of samples

**Data Preprocessing**:
- Converts show IDs and asset types to numerical indices
- Pads sequences shorter than window_size with zeros
- Truncates sequences longer than window_size
- Returns tensors ready for model input

### From `model.py`

#### `RecSysModel` (Class)
**Purpose**: LSTM-based neural network for sequential recommendation.

**Architecture**:
```
Input:
  - Show IDs: (batch_size, window_size)
  - Asset Types: (batch_size, window_size)
  - Watch Minutes: (batch_size, window_size)

Processing:
  1. Embedding layers convert show IDs and asset types to dense vectors
  2. Concatenate embeddings with watch minutes
  3. LSTM processes the sequence
  4. Dropout for regularization
  5. Linear layer projects to output dimension
  
Output: (batch_size, num_shows) - scores for each possible next show
```

**Key Methods**:
- `__init__()`: Initializes embedding layers, LSTM, and output projection
- `forward()`: Performs forward pass through the network
- `init_hidden()`: Creates initial hidden state for LSTM

**Parameters**:
- `num_shows`: Total number of unique shows + 1 (padding)
- `num_asset_types`: Total number of asset types + 1 (padding)
- `show_embedding_dim`: Dimension of show embeddings (32)
- `asset_type_embedding_dim`: Dimension of asset type embeddings (32)
- `hidden_dim`: LSTM hidden state dimension (128)
- `output_dim`: Number of shows to predict (equals num_shows)
- `dropout`: Dropout rate for regularization (0.2)

#### `train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)`
**Purpose**: Trains the recommendation model.

**Training Loop**:
```
For each epoch:
  1. Training Phase:
     - Iterate through training batches
     - Forward pass: model predictions
     - Calculate CrossEntropyLoss
     - Backward pass: compute gradients
     - Update weights with Adam optimizer
     - Track top-10 recall (is target in top 10 predictions?)
  
  2. Validation Phase:
     - Evaluate on validation set (no gradient computation)
     - Calculate validation loss and top-10 recall
     
  3. Print epoch summary with metrics
```

**Metrics Tracked**:
- **Loss**: CrossEntropyLoss (measures prediction confidence)
- **Top-10 Recall**: Percentage of times the actual next show appears in top 10 predictions

**Returns**: `history` dictionary containing:
- `train_loss`: List of training losses per epoch
- `train_10_recall`: List of training top-10 recall per epoch
- `val_loss`: List of validation losses per epoch
- `val_10_recall`: List of validation top-10 recall per epoch

#### `evaluate_model(model, test_loader, device)`
**Purpose**: Evaluates trained model on test set.

**Process**:
1. Sets model to evaluation mode
2. Iterates through test batches without computing gradients
3. Calculates test loss and top-10 recall
4. Prints final test results

**Returns**: Dictionary with:
- `test_loss`: Average loss on test set
- `test_10_recall`: Top-10 recall percentage
- `correct`: Number of correct predictions in top-10
- `total`: Total number of test samples

### From `metrics.py`
Note: These functions are defined but not currently used in `main.py`.

#### `top_k_recall(y_true, y_pred, k=10, average='macro')`
Calculates top-k recall using sklearn's recall_score method.

#### `top_k_precision(y_true, y_pred, k=10, average='macro')`
Calculates top-k precision using sklearn's precision_score method.

#### `top_k_f1(y_true, y_pred, k=10, average='macro')`
Calculates top-k F1 score.

#### `evaluate_top_k_metrics(model, data_loader, k_values=[1,5,10,20], device='cpu')`
Evaluates model with multiple top-k metrics for different k values.

## Data Flow

```
1. Raw Data (playback_sessions.parquet)
   ↓
2. Preprocessing (read_data.py)
   → Creates sequences of shows watched
   → Generates (input_sequence, target) pairs
   → Saves to processed_dataset_window20.pkl
   ↓
3. Data Loading (data_loader.py)
   → Loads processed data
   → Splits into train/val/test
   → Creates PyTorch DataLoaders
   ↓
4. Model Training (main.py + model.py)
   → Initializes RecSysModel
   → Trains with LSTM architecture
   → Validates performance
   ↓
5. Evaluation
   → Tests on held-out test set
   → Measures top-10 recall
```

## Dataset Details

**Source**: `data/playback_sessions.parquet`

**Schema**:
- `user_id`: Unique user identifier (int32)
- `playback_session_id`: Session identifier (int32)
- `show_id`: TV show identifier (int32)
- `asset_type`: Type of content - "CHANNEL", "RECORDING", or "VOD" (string)
- `episode_id`: Episode identifier (int32)
- `day`: Date of viewing (string)
- `time`: Time of viewing (string)
- `watch_minutes`: Duration watched in minutes (int32)

**Dataset Size**: 100,000 users over 30 days

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd show_rec_sys

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas==2.2.3
pyarrow==18.0.0
scikit-learn==1.6.1
matplotlib==3.10.1
seaborn==0.13.2
numpy==1.26.4
jupyter==1.1.1
torch==2.5.1
```

## Usage

### 1. Preprocess the Data (if needed)
```bash
python read_data.py
```
This creates:
- `data/processed_dataset_window20.pkl`: Processed sequences
- `data/all_shows_and_asset_types.pkl`: Show and asset type mappings

### 2. Train the Model
```bash
python main.py
```

**Expected Output**:
```
Using device: cuda
Creating data loaders...
Loaded X samples
Dataset split:
  Train: X samples (70.0%)
  Validation: X samples (10.0%)
  Test: X samples (20.0%)

Initializing model...
Model parameters: XXX,XXX

Starting training...
Epoch 1/5, Batch 0/XX, Loss: X.XXXX
...
Epoch 1/5:
  Train Loss: X.XXXX, Train 10-Recall: XX.XX%
  Val Loss: X.XXXX, Val 10-Recall: XX.XX%
--------------------------------------------------
...
Training completed!
Best 10-Recall: XX.XX%
```

### 3. Evaluate on Test Set
Uncomment lines 60-65 in `main.py` to enable test evaluation:
```python
print(f"\nEvaluating on test set...")
test_results = evaluate_model(model, test_loader, device=device)
print(f"\nTraining completed!")
print(f"Final test accuracy: {test_results['test_accuracy']:.2f}%")
```

## Model Architecture Details

### Input Features
1. **Show ID Sequence**: Last 20 shows watched by the user
2. **Asset Type Sequence**: Type of each viewing (CHANNEL/RECORDING/VOD)
3. **Watch Duration**: Minutes watched for each show

### Model Components
1. **Embedding Layers**:
   - Show Embedding: Maps each show to a 32-dim vector
   - Asset Type Embedding: Maps each asset type to a 32-dim vector

2. **Feature Concatenation**:
   - Combines show embeddings, asset type embeddings, and watch minutes
   - Total input dimension: 32 + 32 + 1 = 65 per timestep

3. **LSTM Layer**:
   - Processes sequential information
   - Hidden dimension: 128
   - Single layer with 20% dropout

4. **Output Layer**:
   - Linear projection from hidden_dim (128) to num_shows
   - Produces logits for each possible next show

### Loss Function
**CrossEntropyLoss**: Treats recommendation as a multi-class classification problem

### Optimizer
**Adam**: Adaptive learning rate optimizer with lr=0.001

## Performance Metrics

### Primary Metric: Top-10 Recall
**Definition**: Percentage of times the actual next show appears in the model's top 10 predictions.

**Why Top-10 Recall?**
- More realistic than top-1 accuracy for recommendation systems
- Users typically see multiple recommendations
- Captures whether the model understands user preferences, even if not perfectly ranked

**Calculation**:
```python
for each prediction:
    top_10_predictions = model.top_k(predictions, k=10)
    if actual_show in top_10_predictions:
        correct += 1
recall = correct / total * 100
```

## Project Structure

```
show_rec_sys/
├── data/
│   ├── playback_sessions.parquet          # Raw viewing data
│   ├── processed_dataset_window20.pkl     # Preprocessed sequences
│   └── all_shows_and_asset_types.pkl      # Vocabulary mappings
├── main.py                                 # Main training script
├── data_loader.py                          # Dataset and data loading utilities
├── model.py                                # LSTM model and training functions
├── metrics.py                              # Evaluation metrics (top-k recall, precision, F1)
├── read_data.py                            # Data preprocessing script
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── LICENSE                                 # License information
```

## Key Design Decisions

### 1. Window Size = 20
- Captures recent viewing behavior
- Balances context and computational efficiency
- Long enough to capture patterns, short enough to avoid sparse data

### 2. LSTM Architecture
- **Why LSTM?** Sequential data requires modeling temporal dependencies
- Captures both short-term and long-term viewing patterns
- Better than simple averaging or last-viewed approaches

### 3. Embedding Dimensions = 32
- Compact representation of shows and asset types
- Reduces parameter count while maintaining expressiveness
- Learns semantic relationships between shows

### 4. Batch Size = 512
- Balances training speed and memory usage
- Provides stable gradient estimates
- Efficient GPU utilization

### 5. Multi-Feature Input
- **Show IDs**: What content was watched
- **Asset Types**: How it was watched (live/recorded/on-demand)
- **Watch Minutes**: Engagement level
- Richer context leads to better predictions

## Next Steps and Improvements

1. **Model Enhancements**:
   - Try deeper LSTM (multiple layers)
   - Experiment with attention mechanisms
   - Add user embeddings for personalization
   - Incorporate time-based features (day of week, time of day)

2. **Training Improvements**:
   - Implement learning rate scheduling
   - Add early stopping based on validation performance
   - Try different optimizers (AdamW, SGD with momentum)
   - Experiment with different loss functions (negative sampling, BPR loss)

3. **Feature Engineering**:
   - Episode information (binge-watching patterns)
   - Show genre/category
   - Show popularity metrics
   - Temporal decay of viewing history

4. **Evaluation**:
   - Implement more sophisticated metrics (NDCG, MAP)
   - Analyze performance by user segments
   - Cold-start user performance
   - Temporal consistency (predict next week)

5. **Production**:
   - Model serving with FastAPI/Flask
   - Batch inference pipeline
   - Model versioning and A/B testing
   - Monitoring and retraining pipelines

## License

See LICENSE file for details.
