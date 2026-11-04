
import json

import torch
from torch.utils.data import Dataset


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


class MazeDataset(Dataset):

    def __init__(self, file_path, vocab_path):
        with open(file_path, "r") as f:
            self.examples = json.load(f)

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        # Convert vocabulary IDs into four classification labels:
        # UP -> 0, DOWN -> 1, LEFT -> 2, RIGHT -> 3
        self.action_to_class = {
            action: index
            for index, action in enumerate(ACTIONS)
        }

        self.action_id_to_class = {
            self.vocab[action]: self.action_to_class[action]
            for action in ACTIONS
        }

        # Fail early if the generated dataset is inconsistent.
        for example in self.examples:
            assert len(example["input_ids"]) == 66
            assert example["target_id"] in self.action_id_to_class

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        input_ids = torch.tensor(
            example["input_ids"],
            dtype=torch.long
        )

        target_class = self.action_id_to_class[
            example["target_id"]
        ]

        target = torch.tensor(target_class, dtype=torch.long)

        return input_ids, target
