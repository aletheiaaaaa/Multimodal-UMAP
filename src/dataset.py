import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, AutoTokenizer, AutoModel

def load_data(dataset: str, text_encoder: str, vision_encoder: str, split: str = "train"):
    data = load_dataset(dataset, split=split)

    text_tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    text_model = AutoModel.from_pretrained(text_encoder)

    image_processor = ViTImageProcessor.from_pretrained(vision_encoder)
    vision_model = ViTForImageClassification.from_pretrained(vision_encoder)

    processed_data = {
        "texts": [],
        "images": []
    }

    for example in data:
        text = example["text"]
        image = example["image"]

        tokenized_text = text_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).input_ids.squeeze(0)
        processed_image = image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        with torch.no_grad():
            text_features = text_model(tokenized_text.unsqueeze(0)).pooler_output
            image_features = vision_model(processed_image.unsqueeze(0)).pooler_output

        processed_data["texts"].append(text_features.squeeze(0))
        processed_data["images"].append(image_features.squeeze(0))

    processed_data["texts"] = torch.stack(processed_data["texts"])
    processed_data["images"] = torch.stack(processed_data["images"])

    return processed_data

def train_test_split(data: dict, test_ratio: float = 0.2):
    num_samples = data["texts"].size(0)
    indices = torch.randperm(num_samples)
    test_size = int(num_samples * test_ratio)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_data = {
        "texts": data["texts"][train_indices],
        "images": data["images"][train_indices]
    }

    test_data = {
        "texts": data["texts"][test_indices],
        "images": data["images"][test_indices]
    }

    return train_data, test_data