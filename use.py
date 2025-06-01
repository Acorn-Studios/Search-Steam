from sentence_transformers import SentenceTransformer
import pickle as pkl
import csv

class SteamTransformer:
    def __init__(self):
        # Load a pretrained Sentence Transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load the embeddings from the pickle file
        self.embeddings = pkl.load(open("embeddings.pkl", "rb"))
        # Load our paired sentences with the appids
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
    def predict(self, prompt):
        embedding = self.encode(prompt)
        similarities = self.similarity(embedding)
        index = similarities.argmax()
        return {"data" : self.labeled[index], "appid": int(self.labeled[index][0]), "url" : "https://store.steampowered.com/app/" + str(self.labeled[index][0])}



# 4. Calculate the embedding similarities
ST = SteamTransformer()
prompt = "This is a test sentence."
print(ST.predict(prompt))