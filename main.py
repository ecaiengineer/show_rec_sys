import torch
import pickle
from data_loader import create_data_loaders
from model import RecSysModel, train_model, evaluate_model

# Configuration
dataset_path = 'data/processed_dataset_window20.pkl'
all_shows_and_asset_types_path = 'data/all_shows_and_asset_types.pkl'
batch_size = 512
num_epochs = 10
learning_rate = 0.001
window_size = 20

with open(all_shows_and_asset_types_path, 'rb') as f:
    all_shows_and_asset_types = pickle.load(f)

all_shows = all_shows_and_asset_types['all_shows']
all_asset_types = all_shows_and_asset_types['all_asset_types']

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create data loaders
print("Creating data loaders...")
train_loader, val_loader, test_loader = create_data_loaders(
    dataset_path=dataset_path,
    all_shows=all_shows,
    all_asset_types=all_asset_types,
    batch_size=batch_size,
    window_size=window_size
)

hyperparameters = {
    'num_shows': len(all_shows)+1,
    'num_asset_types': len(all_asset_types)+1,
    'show_embedding_dim': 128,
    'asset_type_embedding_dim': 128,
    'hidden_dim': 256,
    'output_dim': len(all_shows)+1,
    'dropout': 0.3,
    'num_lstm_layers': 3
}

model_path = f'artifacts/recsys_model_window{window_size}_lstm_{hyperparameters["num_lstm_layers"]}.pth'

# Initialize model
print(f"\nInitializing model...")
model = RecSysModel(
    **hyperparameters
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")



# Train the model
print(f"\nStarting training...")
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device
)

# save the model
checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters
}

torch.save(checkpoint, model_path)

# Evaluate on test set
print(f"\nEvaluating on test set...")
test_results = evaluate_model(model, test_loader, device=device)
print(f"\nTraining completed!")