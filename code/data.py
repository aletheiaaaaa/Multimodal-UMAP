import torch
from pathlib import Path
from scipy import io as sio

def load_labels(dataset: str, subject: int) -> torch.Tensor:
    if dataset not in ["DIR-Wiki", "GOD-Wiki", "ThingsEEG-Text"]:
        raise ValueError(f"Dataset {dataset} is not supported. Choose from 'DIR-Wiki', 'GOD-Wiki', or 'ThingsEEG-Text'.")

    if (dataset == "DIR-Wiki" and not 0 <= subject <= 3) or \
       (dataset == "GOD-Wiki" and not 0 <= subject <= 5) or \
       (dataset == "ThingsEEG-Text" and not 0 <= subject <= 10):
        raise ValueError(f"Subject {subject} is not valid for dataset {dataset}.")

    base = Path("data") / "brain_feature"
    files = [
        str(file) for file in base.rglob(f"sub-{subject:02d}/*.mat") 
        if "ALBERT" not in file.parts 
        and not (file.stem.endswith("_unique"))
    ]

    labels = []
    for file in files:
        contents = sio.loadmat(file)
        labels.append(torch.tensor(contents["class_idx"].squeeze(), dtype=torch.long))

    labels = set(torch.cat(labels, dim=0).tolist())

    return torch.tensor(sorted(labels), dtype=torch.long)

def load_data(dataset: str, mode: str, subject: int) -> torch.Tensor:
    if mode not in ["brain", "textual", "visual"]:
        raise ValueError(f"Mode {mode} is not supported. Choose from 'brain', 'textual', or 'visual'.")

    if dataset not in ["DIR-Wiki", "GOD-Wiki", "ThingsEEG-Text"]:
        raise ValueError(f"Dataset {dataset} is not supported. Choose from 'DIR-Wiki', 'GOD-Wiki', or 'ThingsEEG-Text'.")

    if (dataset == "DIR-Wiki" and not 0 <= subject <= 3) or \
       (dataset == "GOD-Wiki" and not 0 <= subject <= 5) or \
       (dataset == "ThingsEEG-Text" and not 0 <= subject <= 10):
        raise ValueError(f"Subject {subject} is not valid for dataset {dataset}.")

    base = Path("data") / f"{mode}_feature"
    files = [
        str(file) for file in base.rglob(f"sub-{subject:02d}/*.mat") 
        and ("textual_feature" not in file.parts or "GPTNeo" in file.parts)
        and not (file.stem.endswith("_unique"))
    ]

    data = []
    for file in files:
        contents = sio.loadmat(file)
        data.append(torch.tensor(contents["data"], dtype=torch.float32) * (50.0 if mode == "visual" else 2.0) )

    return torch.cat(data, dim=0)

def train_test_split(data: torch.Tensor, train_ratio: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = data.shape[0]
    split_index = int(num_samples * train_ratio)

    return data[:split_index], data[split_index:]