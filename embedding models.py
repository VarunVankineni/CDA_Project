#!/usr/bin/env python
# coding: utf-8



import gensim
from gensim.models.wrappers import FastText
import numpy as np



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
            
            elif path =='glove':
                embeddings_dict = {}
                with open(self.paths[path], 'r', encoding="utf8") as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], "float32")
                        embeddings_dict[word] = vector
                        
                self.models[path] = embeddings_dict
                
    
    def generate_embeddings(self, word, size = 300, subset = True):
        
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
            






if __name__ == '__main__':

    model_paths = {
                    'fast_text':  'cc.en.300',
                    'word2vec': 'GoogleNews-vectors-negative300.bin.gz',
                    'glove': 'glove.6B.300d.txt'
                    }




    embedder = WordEmbedder(model_paths)
    embedder.load_models()
    a = embedder.generate_embeddings('raks', subset = False, size = 120)
    print(a.shape)

