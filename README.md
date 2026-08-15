# Jewellery Visual Search Engine

This project is a visual search engine for jewellery images. It allows the user to upload an image or take a photo using the camera and then displays similar jewellery items.

## How It Works

1. The jewellery images are loaded from the dataset.
2. MobileNetV2 extracts an embedding vector from every image.
3. The embeddings are saved in `embeddings.npy`.
4. FAISS compares the uploaded image with the saved embeddings.
5. Streamlit displays the top similar jewellery images.

## Project Files

* `app.py`: Runs the Streamlit application.
* `prepare_data.py`: Processes the images and creates the embeddings.
* `requirements.txt`: Contains the required Python libraries.
* `data/embeddings.npy`: Contains the saved image embeddings.
* `data/filenames.npy`: Contains the image file paths.
* `data/images`: Contains the jewellery dataset.

## Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Create the embeddings if they have not already been created:

```bash
python prepare_data.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Application Features

* Upload a jewellery image.
* Take a photo using the camera.
* Display the uploaded image.
* Find the top 25 visually similar items.
* Show a message when no similar jewellery is found.
