import os
import numpy as np
import faiss


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embeddings = np.load(os.path.join(BASE_DIR, "data", "embeddings.npy")).astype("float32")
filenames = np.load(os.path.join(BASE_DIR, "data", "filenames.npy"))

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

correct = 0
total = 0

for i in range(len(embeddings)):
    scores, positions = index.search(embeddings[i:i + 1], 6)
    query_class = os.path.basename(os.path.dirname(filenames[i]))

    for position in positions[0][1:6]:
        result_class = os.path.basename(os.path.dirname(filenames[position]))
        if query_class == result_class:
            correct += 1
        total += 1

precision_at_5 = correct / total
print("Precision@5:", round(precision_at_5, 3))
