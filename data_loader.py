import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import random


class RecSysDataset(Dataset):
    """
    PyTorch Dataset for recommendation system data.
    
    Each sample contains:
    - user_id: User identifier
    - show_sequence: Sequence of show IDs watched by the user
    - asset_types: Sequence of asset types (CHANNEL, RECORDING, VOD)
    - watch_minutes: Sequence of watch durations
    - target: Next show ID to predict
    """
    
    def __init__(self, 
            data: List[Dict], 
            all_shows: List[int],
            all_asset_types: List[str], 
            window_size: int = 20
        ):
        
        self.data = data
        self.pad_watch_token = 0
        self.pad_show_token = 0
        self.pad_asset_token = 'UNK'
        self.show_to_idx = {show_id: idx for idx, show_id in enumerate(
            np.concatenate([[self.pad_show_token], all_shows]))} # 0 is for padding shows
        self.idx_to_show = {idx: show_id for show_id, idx in self.show_to_idx.items()}
        self.asset_type_to_idx = {asset_type: idx for idx, asset_type in enumerate(
            np.concatenate([[self.pad_asset_token], all_asset_types]))} # 'UNK' is for padding asset types
        self.window_size = window_size
        self.num_shows = len(self.show_to_idx)
        self.num_asset_types = len(self.asset_type_to_idx)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            tuple: (show_sequence, asset_types, watch_minutes, target)
                - show_sequence: (window_size,) - show IDs
                - asset_types: (window_size,) - asset type indices
                - watch_minutes: (window_size,) - watch durations
                - target: (1,) - target show ID
        """
        sample = self.data[idx]
        
        # Get sequences from the sample
        show_ids = sample['inputs']['show_id']
        asset_types = sample['inputs']['asset_type']
        watch_minutes = sample['inputs']['watch_minutes']
        target = self.show_to_idx.get(sample['target'], self.pad_show_token)
        
        # Convert show IDs to indices
        show_indices = [self.show_to_idx.get(show_id, self.pad_show_token) for show_id in show_ids]
        asset_types_indices = [self.asset_type_to_idx.get(asset_type, self.pad_asset_token) for asset_type in asset_types]
        
        # Pad or truncate sequences to window_size
        if len(show_indices) < self.window_size:
            # Pad with padding tokens
            pad_length = self.window_size - len(show_indices)
            show_indices = [self.show_to_idx.get(self.pad_show_token)] * pad_length + show_indices
            asset_types_indices = [self.asset_type_to_idx.get(self.pad_asset_token)] * pad_length + asset_types_indices
            watch_minutes = np.concatenate((np.zeros(pad_length), watch_minutes))
        
        # Convert to tensors
        show_tensor = torch.tensor(show_indices, dtype=torch.long)
        asset_types_tensor = torch.tensor(asset_types_indices, dtype=torch.long)
        watch_minutes_tensor = torch.tensor(watch_minutes, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return [show_tensor, asset_types_tensor, watch_minutes_tensor], target_tensor


def split_dataset(dataset: List[Dict], test_size: float = 0.2, val_size: float = 0.1, 
                  random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: List of data samples
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test split)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        dataset, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(dataset)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(dataset)*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(dataset)*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_data_loaders(dataset_path: str, 
                       all_shows: List[int],
                       all_asset_types: List[str],
                       batch_size: int = 32, 
                       window_size: int = 20, test_size: float = 0.2, 
                       val_size: float = 0.1, num_workers: int = 0, 
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_path: Path to the pickle file containing the dataset
        batch_size: Batch size for data loaders
        window_size: Window size for sequences
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        num_workers: Number of worker processes for data loading
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, metadata)
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} samples")

    # Split dataset
    train_data, val_data, test_data = split_dataset(
        dataset, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Create datasets
    train_dataset = RecSysDataset(train_data, all_shows, all_asset_types, window_size)
    val_dataset = RecSysDataset(val_data, all_shows, all_asset_types, window_size)
    test_dataset = RecSysDataset(test_data, all_shows, all_asset_types, window_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    
    print(f"Data loaders created successfully!")
    print(f"  Batch size: {batch_size}")
    print(f"  Window size: {window_size}")
    
    return train_loader, val_loader, test_loader