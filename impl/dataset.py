import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision.transforms import functional as TF
from tqdm import tqdm

def load_data(split: str) -> dict:
    """Load and preprocess Flickr30k dataset with cached feature extraction.

    Preprocesses text and images. Results are cached to data/{split}_data.pt.

    Args:
        split: Dataset split to load (e.g., "train", "test").

    Returns:
        Dictionary with keys "texts" and "images", each containing a tensor
        of shape (N, D) where N is the number of samples.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(f"data/{split}_data.pt"):
        return torch.load(f"data/{split}_data.pt", map_location=device_str)

    data = load_dataset("AnyModal/flickr30k", split=split, streaming=True)
    batches = data.batch(batch_size=128 if torch.cuda.is_available() else 8)

    device = torch.device(device_str)

    text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    image_size = 32
    
    vocab = set()
    texts = []
    images = []

    # First pass: build vocabulary
    for batch in tqdm(data, desc=f"Building vocabulary for {split}"):
        text = [t[0] for t in batch["alt_text"]]
        tokens = text_tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        for token_ids in tokens["input_ids"]:
            vocab.update(token_ids.tolist())

    vocab = sorted(list(vocab))
    vocab_size = len(vocab)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    # Second pass: extract features
    for batch in tqdm(batches, desc=f"Loading {split} data"):
        text = [t[0] for t in batch["alt_text"]]
        image_list = batch["image"]

        encoded_input = text_tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        text_features = torch.zeros(len(text), vocab_size, device=device)
        for i, token_ids in enumerate(encoded_input["input_ids"]):
            for token_id in token_ids:
                if token_id.item() in token_to_idx:
                    text_features[i, token_to_idx[token_id.item()]] += 1
        texts.append(text_features)

        image_tensors = [TF.resize(TF.to_tensor(img.convert("RGB")), [image_size, image_size]) for img in image_list]
        image_features = torch.stack(image_tensors).flatten(1).to(device)
        images.append(image_features)

    data_dict = {
        "texts": torch.cat(texts, dim=0),
        "images": torch.cat(images, dim=0)
    }

    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(data_dict, f"data/{split}_data.pt")

    return data_dict