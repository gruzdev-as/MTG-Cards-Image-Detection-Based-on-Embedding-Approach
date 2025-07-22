import os

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import CLIPModel, CLIPProcessor

load_dotenv()  # Load variables from .env
token = os.getenv("HF_TOKEN")


class EmbeddingGenerator:
    """Generate cropped image embedding using CLIP."""

    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", token=token)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", token=token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("Model has loaded")

    def generate_image_embedding(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Use CLIP to generate embedding vectors.

        Args:
            images (List[np.ndarray] | np.ndarray): The warped image or list of warped images

        Returns:
            np.array: Generated embedding or list of generated embeddings

        """
        if isinstance(images, (list, np.ndarray)):
            inputs = self.processor(images=images, return_tensors="pt")
        else:
            inputs = self.processor(images=[images], return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features.cpu().numpy()
