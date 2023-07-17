# %% create embedding
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd

sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to('cuda:6')

with open("description.csv", "r") as f:
    lines = f.readlines()

lines = [line.strip().split(",") for line in lines]

for idx, line in enumerate(lines):
    lines[idx] = [line[0], ",".join(line[1:])]

df = pd.DataFrame(lines, columns=["id", "desc"])
embeddings = sentence_model.encode(df["desc"].tolist(), show_progress_bar=True)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print(embeddings.shape)
# %%
