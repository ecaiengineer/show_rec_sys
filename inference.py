import torch
import numpy as np
from typing import List
from model import RecSysModel


class LSTMRecommender:
    """
    A class to make predictions for a given user based on their viewing history.
    """
    def __init__(self,
                 check_point_file: str = None,
                 device: str = "cpu"):

        """
        Initialize the LSTMRecommender class.
        
        Args:
            check_point_file (str): Path to the check point file
            device (str): Device to run the model on
        """
        self.device = device
        self.pad_show_token = 0
        self.pad_asset_token = 'UNK'
        self.minutes_pad_token = 0
        self.window_size = 20

        try:
            self._checkpoint = torch.load(check_point_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Check point file not found at {check_point_file}")

        # Initialize model with saved hyperparameters
        self._inference_dict = self._checkpoint['inference_dict']
        self.model = RecSysModel(**self._checkpoint['hyperparameters'])
        self.model.load_state_dict(self._checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

         # create mappings
        all_shows = np.arange(1, 13350, dtype=np.int32)
        all_asset_types = np.array(['VOD', 'RECORDING', 'CHANNEL', 'LOOKBACK'], dtype=object)

        self.show_to_idx = {show_id: idx for idx, show_id in enumerate(
            np.concatenate([[self.pad_show_token], all_shows]))} # 0 is for padding shows
        self.idx_to_show = {idx: show_id for show_id, idx in self.show_to_idx.items()}
        self.asset_type_to_idx = {asset_type: idx for idx, asset_type in enumerate(
            np.concatenate([[self.pad_asset_token], all_asset_types]))} # 'UNK' is for padding asset types

    def predict_shows(self, user_id: int) -> List[int]:
        """
        Predict ranked list of shows for a given user based on their viewing history.
        """
        
        # look up user in inference dataset dict
        user_data = self._inference_dict[user_id]

        if user_data is None:
            raise ValueError(f"User ID {user_id} not found in inference dataset")
        
        # Prepare input data
        show_ids = user_data['inputs']['show_id']
        asset_types = user_data['inputs']['asset_type']
        watch_minutes = user_data['inputs']['watch_minutes']

        if len(show_ids) < self.window_size:
            # pad with padding tokens, preceding tokens are pad tokens
            pad_length = self.window_size - len(show_ids)
            show_ids = np.concatenate([[self.pad_show_token] * pad_length, show_ids])
            asset_types = np.concatenate([[self.pad_asset_token] * pad_length, asset_types])
            watch_minutes = np.concatenate([[self.minutes_pad_token] * pad_length, watch_minutes])

        # Convert to indices
        show_indices = [self.show_to_idx.get(show_id, self.pad_show_token) for show_id in show_ids]
        asset_type_indices = [self.asset_type_to_idx.get(asset_type, self.pad_asset_token) for asset_type in asset_types]
        
        # Convert to tensors and add batch dimension
        show_tensor = torch.tensor(show_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        asset_type_tensor = torch.tensor(asset_type_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        watch_minutes_tensor = torch.tensor(watch_minutes, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs, _ = self.model(show_tensor, asset_type_tensor, watch_minutes_tensor)
            
            # Get ranked indices
            ranked_indices = torch.argsort(outputs.squeeze(), descending=True)

        # Convert indices back to show IDs
        ranked_shows = [self.idx_to_show.get(idx, self.pad_show_token) for idx in ranked_indices.cpu().tolist()]
        
        return ranked_shows


if __name__ == "__main__":
    recommender = LSTMRecommender(
        check_point_file='artifacts/lstm_2.pth'
    )
    user_id = 789
    ranked_shows = recommender.predict_shows(user_id=user_id)
    print(f"first 10 ranked shows for user {user_id}: {ranked_shows[:10]}")