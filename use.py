from sentence_transformers import SentenceTransformer
import pickle as pkl
import csv

class SteamTransformer:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = pkl.load(open("embeddings.pkl", "rb"))
        csv_file = "descriptions_cleaned.csv"
        with open (csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.labeled = [row for row in reader]

    def similarity(self, prompt):
        return self.model.similarity(prompt, self.embeddings,)
    def encode(self, sentence):
        return self.model.encode(sentence, convert_to_tensor=True)
    def get_labeled(self):
        return self.labeled
    def predict(self, prompt, depth=0):
        embedding = self.encode(prompt)
        similarities = self.similarity(embedding)
        sorted_indices = similarities.argsort()[::-1]
        index = sorted_indices[depth]
        return {"appid": int(self.labeled[index][0]), "url" : "https://store.steampowered.com/app/" + str(self.labeled[index][0]), "similar_row" : self.labeled[index]}