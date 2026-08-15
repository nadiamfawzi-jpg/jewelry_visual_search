# Jewellery Visual Search Engine

This project uses pretrained MobileNetV2 to turn jewellery images into embedding vectors. FAISS compares a query vector with the saved catalog vectors, and Streamlit displays the most similar products.

## Project structure

```text
jewelry_search/
├── data/
│   ├── images/
│   ├── embeddings.npy
│   └── filenames.npy
├── prepare_data.py
├── evaluate.py
├── app.py
└── requirements.txt
```

## Run the project

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
python prepare_data.py
python evaluate.py
streamlit run app.py
```

Run `prepare_data.py` only when catalog images change. It creates `data/embeddings.npy` and `data/filenames.npy`. Both files must be uploaded to GitHub before deploying on Streamlit Community Cloud.

## Streamlit Cloud fix

The error `No such file or directory: data/embeddings.npy` means the embeddings were not generated or were not committed to GitHub. This solution also uses absolute paths based on `app.py`, so it does not depend on Streamlit's working directory.

## Discussion answers

### Semantic gap

A silver ring and a gold ring with the same setting can be visually similar in shape but different in color. Whether it is relevant depends on what the user wants. A future version could combine visual similarity with filters such as item type, metal, and color.

### Improving Precision@5

Precision can be improved by adding more varied training images, fine-tuning MobileNetV2 on jewellery, using a stronger pretrained model, improving image quality, or training with triplet loss. The simplest first improvement is to fine-tune the model using labelled jewellery images.

