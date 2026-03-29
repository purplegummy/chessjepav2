
import torch

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path, weights_only=True)
        self.data["states"]      = self.data["states"].float()
        self.data["next_states"] = self.data["next_states"].float()
        self.data["actions"]     = self.data["actions"].long()
        self.has_next_evals = "next_evals" in self.data
        self.has_evals      = "evals"      in self.data

    def __len__(self) -> int:
        return len(self.data["actions"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "state":      self.data["states"][idx],      # (17, 8, 8) float
            "next_state": self.data["next_states"][idx], # (17, 8, 8) float
            "action":     self.data["actions"][idx],     # int64 scalar
            "result":     self.data["results"][idx],     # int8 scalar (-1, 0, 1)
        }
        if self.has_evals:
            item["eval"] = self.data["evals"][idx]            # int16 scalar
        if self.has_next_evals:
            item["next_eval"] = self.data["next_evals"][idx]  # int16 scalar
        return item