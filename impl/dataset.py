import torch
import os
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
from tqdm import tqdm

def load_data(split: str) -> dict:
    if os.path.exists(f"data/{split}_data.pt"):
        return torch.load(f"data/{split}_data.pt")

    data = load_dataset("AnyModal/flickr30k", split=split, streaming=True)
    batches = data.batch(batch_size=128 if torch.cuda.is_available() else 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)

    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    texts = []
    images = []

    for batch in tqdm(batches, desc=f"Loading {split} data"):
        text = [t[0] for t in batch["alt_text"]]
        image_list = batch["image"]

        encoded_input = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = text_model(**encoded_input).pooler_output
        texts.append(text_features)

        processed_images = torch.stack([image_transform(img) for img in image_list]).to(device)
        with torch.no_grad():
            image_features = image_model.encode(processed_images).latent_dist.mean
        images.append(image_features)

    data_dict = {
        "texts": torch.cat(texts, dim=0),
        "images": torch.cat(images, dim=0)
    }

    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(data_dict, f"data/{split}_data.pt")

    return data_dict