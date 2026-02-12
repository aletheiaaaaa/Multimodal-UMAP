import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from tqdm import tqdm

def load_data(split: str) -> dict:
    """Load and preprocess Flickr30k dataset with cached feature extraction.

    Extracts text features using BERT (pooler output) and image features using
    Stable Diffusion VAE latents. Results are cached to data/{split}_data.pt.

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
    text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    texts = []
    images = []

    for batch in tqdm(batches, desc=f"Loading {split} data"):
        text = [t[0] for t in batch["alt_text"]]
        image_list = batch["image"]

        encoded_input = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = text_model(**encoded_input).pooler_output
        texts.append(text_features)

        processed_images = image_processor(images=image_list, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = image_model(**processed_images).last_hidden_state[:, 0]
        images.append(image_features)

    data_dict = {
        "texts": torch.cat(texts, dim=0),
        "images": torch.cat(images, dim=0)
    }

    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(data_dict, f"data/{split}_data.pt")

    return data_dict