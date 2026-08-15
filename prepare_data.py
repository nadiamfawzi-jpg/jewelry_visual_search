import os
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "images")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
FILENAMES_PATH = os.path.join(BASE_DIR, "data", "filenames.npy")


# Load MobileNetV2 without its final classification layer.
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier = torch.nn.Identity()
model.eval()
preprocess = weights.transforms()


filenames = []
embeddings = []

for folder_name in sorted(os.listdir(IMAGE_FOLDER)):
    folder_path = os.path.join(IMAGE_FOLDER, folder_name)

    if os.path.isdir(folder_path):
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, file_name)

                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = preprocess(image).unsqueeze(0)

                    with torch.no_grad():
                        embedding = model(image_tensor).numpy()[0]

                    # Normalize so FAISS returns cosine similarity.
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    filenames.append(os.path.relpath(image_path, BASE_DIR))
                except Exception:
                    print("Skipped:", image_path)


embeddings = np.array(embeddings).astype("float32")
np.save(EMBEDDINGS_PATH, embeddings)
np.save(FILENAMES_PATH, np.array(filenames))

print("Images processed:", len(filenames))
print("Embeddings shape:", embeddings.shape)
print("Saved in the data folder")
