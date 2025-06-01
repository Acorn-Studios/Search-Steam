from sentence_transformers import SentenceTransformer
import csv
import pickle as pkl


# 1. Load a pretrained Sentence Transformer model
print("[TRAIN.PY] Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
print("[TRAIN.PY] Loading data...")
csv_file = "descriptions_cleaned.csv"
with open (csv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    labeled = [row for row in reader]

input = [i[1] for i in labeled] # Input is only the second column of the CSV

# 2. Calculate embeddings by calling model.encode()
print("[TRAIN.PY] Calculating embeddings...")
embeddings = model.encode(input, show_progress_bar=True, batch_size=256, convert_to_tensor=True)

# 3. Save our embeddings
pkl.dump(embeddings, open("embeddings.pkl", "wb"))
print("[TRAIN.PY] Embeddings saved to embeddings.pkl")