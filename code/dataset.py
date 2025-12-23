import torch
from pathlib import Path
from scipy import io as sio

def load_labels(subject: int, data_dir: str) -> torch.Tensor:
    if not 1 <= subject <= 10:
        raise ValueError(f"Subject {subject} is not valid for dataset.")

    base = Path(data_dir).expanduser() / "ThingsEEG-Text" / "brain_feature"
    files = [
        str(file) for file in base.rglob(f"sub-{subject:02d}/*.mat") if not file.stem.endswith("_unique")
    ]

    labels = []
    for file in files:
        contents = sio.loadmat(file)
        labels.append(torch.tensor(contents["class_idx"].squeeze(), dtype=torch.long))

    labels = set(torch.cat(labels, dim=0).tolist())

    return torch.tensor(sorted(labels), dtype=torch.long)

def load_data(mode: str, subject: int, data_dir: str) -> torch.Tensor:
    if mode not in ["brain", "textual", "visual"]:
        raise ValueError(f"Mode {mode} is not supported. Choose from 'brain', 'textual', or 'visual'.")

    if not 1 <= subject <= 10:
        raise ValueError(f"Subject {subject} is not valid for dataset.")

    base = Path(data_dir).expanduser() / "ThingsEEG-Text" / f"{mode}_feature"
    files = [
        str(file) for file in base.rglob(f"sub-{subject:02d}/*.mat") if not (file.stem.endswith("_unique"))
    ]

    data = []
    for file in files:
        contents = sio.loadmat(file)
        data.append(torch.tensor(contents["data"], dtype=torch.float32) * (50.0 if mode == "visual" else 2.0) )

    data = torch.cat(data, dim=0)
    return data.reshape(data.shape[0], -1)

def train_test_split(data: torch.Tensor, train_ratio: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = data.shape[0]
    split_index = int(num_samples * train_ratio)

    return data[:split_index], data[split_index:]