import torch
import os
import pickle
from torchvision import transforms
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def load_data(split: str) -> dict:
    """Load and preprocess Flickr30k dataset with cached feature extraction.

    Extracts text features using TF-IDF and images as resized pixel tensors.
    Results are cached to data/{split}_data.pt.

    Args:
        split: Dataset split to load (e.g., "train", "test").

    Returns:
        Dictionary with keys "texts" and "images", each containing a tensor
        of shape (N, D) where N is the number of samples.
    """
    if os.path.exists(f"data/{split}_data.pt"):
        return torch.load(f"data/{split}_data.pt", map_location="cpu")

    if not os.path.exists("data"):
        os.makedirs("data")

    data = load_dataset("AnyModal/flickr30k", split=split, streaming=True)
    batches = data.batch(batch_size=128)

    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])

    captions = []
    images = []

    for batch in tqdm(batches, desc=f"Loading {split} data"):
        captions.extend([t[0] for t in batch["alt_text"]])

        image_list = batch["image"]
        processed_images = torch.stack([image_transform(img) for img in image_list])
        images.append(processed_images.flatten(start_dim=1))

    vectorizer_path = "data/tfidf_vectorizer.pkl"
    if split == "train":
        vectorizer = TfidfVectorizer(max_features=4096)
        text_features = vectorizer.fit_transform(captions)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
    else:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        text_features = vectorizer.transform(captions)

    data_dict = {
        "texts": torch.tensor(text_features.toarray(), dtype=torch.float32),
        "images": torch.cat(images, dim=0)
    }

    torch.save(data_dict, f"data/{split}_data.pt")

    return data_dict
