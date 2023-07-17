# %% app
import gradio as gr
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to('cuda:6')
with open("embeddings.pkl", 'rb') as f:
    embeddings = pickle.load(f)
embeddings = embeddings.astype("float32")

embedding_size = embeddings.shape[1]
n_clusters = 5
num_results = 5

quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT,)

index.train(embeddings)
index.add(embeddings)


def _search(query):
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding.reshape(1, -1)
    distance, indices = index.search(query_embedding, num_results)
    images = [f"images/{i}.jpg" for i in indices[0] if i != -1]

    return images

with gr.Blocks() as demo:

    query = gr.Textbox(lines=1, label="search query")
    outputs=gr.Gallery(preview=True)
    submit = gr.Button(value="search")
    submit.click(_search, inputs=query, outputs=outputs)

demo.launch()

# %%
