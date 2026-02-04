import torch
import os
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, AutoTokenizer, AutoModel
from tqdm import tqdm

def load_data(split: str) -> dict:
    if os.path.exists(f"/data/{split}_data.pt"):
        return torch.load(f"/data/{split}_data.pt")

    data = load_dataset("AnyModal/flickr30k", split=split, streaming=True)
    batches = data.batch(batch_size=64)

    text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vision_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    texts = []
    images = []

    for batch in tqdm(batches, desc=f"Loading {split} data"):
        text = [t[0] for t in batch["alt_text"]]
        image = batch["image"]

        encoded_input = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = text_model(**encoded_input).last_hidden_state.mean(dim=1)
        texts.append(text_features.squeeze(0))

        processed_image = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = vision_model(**processed_image).logits
        images.append(image_features.squeeze(0))

    data_dict = {
        "texts": torch.cat(texts, dim=0),
        "images": torch.cat(images, dim=0)
    }

    torch.save(data_dict, f"/data/{split}_data.pt")

    return data_dict