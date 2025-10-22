import torch
import torch.nn as nn


class RecSysModel(nn.Module):
    """
    A recommendation system model using LSTM for sequential recommendation.
    
    This model takes sequential show watch history data and predicts the next show
    the user is likely to watch.
    """
    
    def __init__(self, 
                num_shows,
                num_asset_types,
                show_embedding_dim, 
                asset_type_embedding_dim, 
                hidden_dim, 
                output_dim, 
                dropout=0.2, 
                sequence_length=None
        ):
        """
        Initialize the RecSysModel with LSTM architecture.
        
        Args:
            show_embedding_dim (int): Dimension of show embedding
            asset_type_embedding_dim (int): Dimension of asset type embedding
            hidden_dim (int): Hidden dimension of LSTM layers
            output_dim (int): Dimension of output (e.g., number of shows to recommend)
            dropout (float): Dropout rate for LSTM layers (default: 0.2)
        """
        super(RecSysModel, self).__init__()
        self.num_shows = num_shows
        self.num_asset_types = num_asset_types
        self.show_embedding_dim = show_embedding_dim
        self.asset_type_embedding_dim = asset_type_embedding_dim
        self.input_dim = self.show_embedding_dim + self.asset_type_embedding_dim + 1 # + 1 for watch minutes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.sequence_length = sequence_length

        # Embedding layers
        self.embedding_show = nn.Embedding(num_shows, show_embedding_dim)
        self.embedding_asset_type = nn.Embedding(num_asset_types, asset_type_embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, show_ids, asset_types, watch_minutes, hidden=None):
        """
        Forward pass through the LSTM network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden (tuple, optional): Hidden state (h_0, c_0) for LSTM
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim)
            tuple: Final hidden state (h_n, c_n)
        """

        # Embed show IDs
        show_embeddings = self.embedding_show(show_ids)
        asset_type_embeddings = self.embedding_asset_type(asset_types)
        

        # Concatenate embeddings
        inp = torch.cat([show_embeddings, asset_type_embeddings, watch_minutes.unsqueeze(2)], dim=2)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(inp, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to output dimension
        output = self.output_layer(lstm_out)
        #output = self.softmax(output[:, -1, :])
        return output[:, -1, :], hidden
    
    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size (int): Batch size
            device (torch.device, optional): Device to create tensors on
        
        Returns:
            tuple: Initial hidden state (h_0, c_0)
        """
        if device is None:
            device = next(self.parameters()).device
            
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        return (h_0, c_0)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device='cpu'):
    """
    Train the RecSysModel using cross entropy loss for multiclass classification.
    
    Args:
        model (RecSysModel): The recommendation model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
    
    Returns:
        dict: Training history containing losses and accuracies
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_10_recall': [],
        'val_loss': [],
        'val_10_recall': []
    }
    
    print(f"Starting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Move data to device
            show_ids, asset_types, watch_minutes = sequences
            show_ids = show_ids.to(device)
            asset_types = asset_types.to(device)
            watch_minutes = watch_minutes.to(device)
            targets = targets.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions, _ = model(show_ids, asset_types, watch_minutes)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            _, predicted_top_10 = torch.topk(predictions.data, 10, dim=1)
            train_total += targets.size(0)
            for i in range(targets.size(0)):
                train_correct += (predicted_top_10[i] == targets[i]).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_10_recall = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                show_ids, asset_types, watch_minutes = sequences    
                asset_types = asset_types.to(device)
                watch_minutes = watch_minutes.to(device)
                targets = targets.to(device)
                
                predictions, _ = model(show_ids, asset_types, watch_minutes)
                
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                _, predicted_top_10 = torch.topk(predictions.data, 10, dim=1)
                val_total += targets.size(0)
                for i in range(targets.size(0)):
                    val_correct += (predicted_top_10[i] == targets[i]).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_10_recall = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_10_recall'].append(train_10_recall)
        history['val_loss'].append(avg_val_loss)
        history['val_10_recall'].append(val_10_recall)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train 10-Recall: {train_10_recall:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val 10-Recall: {val_10_recall:.2f}%')
        print('-' * 50)
    
    print('Training completed!')
    print(f'Best 10-Recall: {max(history["val_10_recall"]):.2f}%')
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained model on test data.
    
    Args:
        model (RecSysModel): The trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to evaluate on
    
    Returns:
        dict: Evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            show_ids, asset_types, watch_minutes = sequences
            show_ids = show_ids.to(device)
            asset_types = asset_types.to(device)
            watch_minutes = watch_minutes.to(device)
            targets = targets.to(device)
            
            predictions, _ = model(show_ids, asset_types, watch_minutes)
        
            loss = criterion(predictions, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(predictions.data, 1)
            _, predicted_top_10 = torch.topk(predictions.data, 10, dim=1)
            total += targets.size(0)
            for i in range(targets.size(0)):
                correct += (predicted_top_10[i] == targets[i]).sum().item()
    
    test_10_recall = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Results:')
    print(f'  Loss: {avg_loss:.4f}')
    print(f'  Recall: {test_10_recall:.2f}%')
    
    return {
        'test_loss': avg_loss,
        'test_10_recall': test_10_recall,
        'correct': correct,
        'total': total
    }


