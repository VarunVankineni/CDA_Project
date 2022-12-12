import pandas as pd
import spacy
import nltk
import matplotlib
import re
import gensim
from gensim.models.wrappers import FastText
import numpy as np
import random
from scipy.spatial.distance import cdist
import datetime

class WordEmbedder():
    def __init__(self, model_paths):
        self.paths = model_paths
        self.models = {}

    def load_models(self):
        for path in self.paths.keys():
            if path == 'fast_text':
                self.models[path] = FastText.load_fasttext_format(self.paths[path])

            elif path == 'word2vec':
                self.models[path] = gensim.models.KeyedVectors.load_word2vec_format(self.paths[path], binary=True)

            elif path == 'glove':
                embeddings_dict = {}
                with open(self.paths[path], 'r', encoding="utf8") as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], "float32")
                        embeddings_dict[word] = vector

                self.models[path] = embeddings_dict

    def generate_embeddings(self, word, size=300, subset=True):

        size = min(size, 300)

        vector_embeddings = []

        for model in self.models.keys():
            try:
                vector_embeddings.append(self.models[model][word][:size])
            except KeyError:
                print('word embedding for {} not found in {}'.format(word, model))
            except:
                print("unknown error")

        if (len(vector_embeddings) != len(self.models.values())) and subset == False:
            vector_embeddings = None

        return np.array(vector_embeddings)

model_paths = {
   # 'fast_text': 'cc.en.300',
    'word2vec': 'GoogleNews-vectors-negative300.bin.gz',
  #  'glove': 'glove.6B.300d.txt'
}

embedder = WordEmbedder(model_paths)
embedder.load_models()

class Board():
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.wordCorpus = self.loadCorpus(model = self.embed_model)
        self.words = self.select25Words()
        self.assignment, self.teamScore = self.assignColors(self.words)
        self.words = pd.Series(list(self.assignment.keys()), name="lemma")
        self.embeds = self.getEmbeddings()
        self.clue, self.dists = self.findClue()

    def loadCorpus(self, model):
        df = pd.Series([k for k, v in model.models['word2vec'].vocab.items() if len(k) > 2], name="lemma")
        df = df[~df.str.contains("[\W\_]")].drop_duplicates()
        df = pd.DataFrame(df)
        return df

    def __checkEmbedding(self, word):
        try:
            _ = self.embed_model.generate_embeddings(word)
            return True
        except:
            return False

    def select25Words(self, use_actual = True):
        if(use_actual):
            with open("basicwords.txt", "r", encoding='utf-8', errors='ignore') as file:
                df = file.readlines()
                df = pd.DataFrame(df, columns=["lemma"])
                df["lemma"] = df.lemma.str.replace("\s+","").str.lower()
                df = df[df["lemma"].apply(self.__checkEmbedding)]
                return df.lemma.sample(25)
        return self.wordCorpus.lemma.sample(25)

    def assignColors(self, words):
        red_first = 0
        color_count = {
            "red":8+red_first,
            "blue":8+(1-red_first),
            "black":1,
            "white":7,
        }
        assign = [[color]*v for color, v in color_count.items()]
        assign = [j for i in assign for j in i]
        random.shuffle(assign)
        mapping = dict(zip(words,assign))
        teamScore = {
            "red": {"score": 0, "req":color_count["red"], "black":0},
            "blue": {"score": 0, "req":color_count["blue"], "black": 0}
        }
        return mapping, teamScore

    def getEmbeddings(self):
        words, embed_model = self.words, self.embed_model
        embeds = {w:embed_model.generate_embeddings(w, subset=False, size=300) for w in words}
        return embeds

    def similarityScores(self, word, board, model, embeds):
        word = model.generate_embeddings(word, subset=False, size=300)
        dists = cdist(word, np.concatenate([e for _, e in embeds.items()])).flatten()
        dist_dict = {}
        for i, (w, c) in enumerate(board.items()):
            dist_dict[c] = dist_dict.get(c, []) + [dists[i]]
        return dist_dict, dists

    def objective(self, dist_dict, good_treshold, black_treshold,  match_limit, similarity = 0):
        bad_words_limit = dist_dict["red"] + dist_dict["white"] + dist_dict["black"]
        bad_words_limit = min(bad_words_limit) if not similarity else max(bad_words_limit)
        if not similarity:
            good_words = [i for i in dist_dict["blue"] if i < bad_words_limit]
        else:
            good_words = [i for i in dist_dict["blue"] if i > bad_words_limit]
        if good_words:
            good_words_limit = max(good_words) if not similarity else min(good_words)
        else:
            good_words_limit = bad_words_limit
        good_gap =  (bad_words_limit - good_words_limit) * (-1 if similarity else 1)
        black_gap = (dist_dict["black"][0] - good_words_limit) * (-1 if similarity else 1)
        if good_gap<good_treshold or black_gap<black_treshold or good_words_limit>match_limit:
            obj = -1
        else:
            obj = len(good_words)
        return obj

    def getScores(self, word,  good_treshold = 0.15, black_treshold = 0.5, match_limit = 4):
        board, embeds, model, team = self.assignment,  self.embeds, self.embed_model, self.teamScore
        dist_dict, dists = self.similarityScores(word, board, model, embeds)
        obj = self.objective(dist_dict, good_treshold, black_treshold,  match_limit)
        return obj, dists

    def updateClue(self, word):
        scores = self.getScores(word)
        self.clue, self.dists = [word, scores[0]], scores

    def pBoard(self):
        return list(board.assignment.keys())

    def findings(self):
        res = {w: (c, self.dists[1][i]) for i, (w, c) in enumerate(self.assignment.items())}
        res = dict(sorted(res.items(), key=lambda x: x[1][1]))
        return res

    def findClue(self):
        best_word, best_score = "", 0
        for word in self.wordCorpus.lemma:
            wordl = word.lower()
            if word != wordl:
                continue
            if sum([wordl in i or i in wordl for i,_ in self.assignment.items()])>0 or wordl in self.assignment or len(word)<=2:
                continue
            score, _ = self.getScores(word)
            if score>best_score:
                best_word = word
                best_score = score
        scores = self.getScores(best_word)
        suggestion = [best_word, scores[0]]
        return suggestion, scores

    def selectWords(self, words):
        results = {}
        saver = np.array([self.clue, self.findings(), words])
        with open(re.sub("\W", "_", f'ClueTest{str(datetime.datetime.now())[:-7]}.npy'), 'wb') as f:
            np.save(f, saver)

        for w in words:
            if w not in self.assignment:
                raise ValueError
            color = self.assignment[w]
            del self.assignment[w]
            del self.embeds[w]
            results[w] = color
            if color == "blue":
                self.teamScore["blue"]["score"] += 1
            elif color == "black":
                self.teamScore["blue"]["black"] = 1
                break
            else:
                break
        unselected = [w for w in words if w not in results]

        return results, unselected


board = Board(embedder)
board.clue
print(board.findings())


def compareWords(w1,w2, metric = "euclidean"):
    return cdist(embedder.generate_embeddings(w1, subset=False, size=300),
                 embedder.generate_embeddings(w2, subset=False, size=300), metric = metric )
