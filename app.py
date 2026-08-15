import os
import numpy as np
import streamlit as st
import torch
import faiss
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
FILENAMES_PATH = os.path.join(BASE_DIR, "data", "filenames.npy")


st.set_page_config(page_title="Jewellery Visual Search", page_icon="💎", layout="wide")


@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


@st.cache_resource
def load_index():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    filenames = np.load(FILENAMES_PATH)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, filenames


def extract_embedding(image, model, preprocess):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor).numpy().astype("float32")

    faiss.normalize_L2(embedding)
    return embedding


st.title("💎 Jewellery Visual Search Engine")
st.write("Upload a jewellery photo or use your camera to find similar products.")


if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(FILENAMES_PATH):
    st.error("The search files are missing. Run `python prepare_data.py`, then upload the data folder to GitHub.")
    st.stop()


model, preprocess = load_model()
index, filenames = load_index()

input_method = st.radio("Choose an image source", ["Upload", "Camera"], horizontal=True)

if input_method == "Upload":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

threshold = st.slider("Minimum similarity", 0.0, 1.0, 0.45, 0.05)


if image_file is not None:
    query_image = Image.open(image_file)
    st.subheader("Your image")
    st.image(query_image, width=250)

    query_embedding = extract_embedding(query_image, model, preprocess)
    scores, positions = index.search(query_embedding, 25)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position != -1 and score >= threshold:
            results.append((float(score), filenames[position]))

    if len(results) == 0:
        st.warning("No similar jewellery was found. Try another image or lower the threshold.")
    else:
        st.subheader("Top similar items")
        columns = st.columns(5)

        for i, (score, file_name) in enumerate(results):
            image_path = os.path.join(BASE_DIR, file_name)
            with columns[i % 5]:
                st.image(image_path, use_container_width=True)
                st.caption(f"Similarity: {score:.2f}")
