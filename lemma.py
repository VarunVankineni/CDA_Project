#%%
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
import re

df = pd.read_csv('tcc_ceds_music.csv')  
df = df[['artist_name','track_name','genre','lyrics','topic']]
df = df.reset_index (drop = True)        
df['sentences'] = df['artist_name']+" "+df['track_name']+" "+df['genre']+" "+df['lyrics']+" "+df['topic']
df = df[['sentences']].drop_duplicates()
#%%
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


music_words = []
pos = []
for sentence in df['sentences']:
    word_list = nltk.word_tokenize(sentence)
    x = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list]
    p = [get_wordnet_pos(w) for w in word_list]
    music_words.append(x)
    pos.append(p)
#%%
x = []
y = []
for i in music_words:
    for j in i:
        x.append(j)
for k in pos:
    for l in k:
        y.append(l) 


df = pd.DataFrame(list(zip(x, y)),
               columns =['lemma', 'PoS'])
df = df[df["lemma"].str.len() >= 3]
df = df[df["PoS"].str.contains("n|a")]
df = df[["lemma"]].drop_duplicates()
df.to_csv("Musicwords.txt", index = False, header = False) 

#%%
with open("movie_lines.txt", "r", encoding='utf-8', errors='ignore') as file:
            df = file.readlines()
df = [re.sub("\n", "", i).split("+++$+++") for i in df]
df = [i for i in df if len(i) == 5]
df = pd.DataFrame(df)
df.columns =['lineID', 'characterID ', 'movieID', 'character name','words']
df = df[['words']]
#%%
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


movie_words = []
pos = []
for sentence in df['words']:
    word_list = nltk.word_tokenize(sentence)
    x = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list]
    p = [get_wordnet_pos(w) for w in word_list]
    movie_words.append(x)
    pos.append(p)
#%%
x = []
y = []
for i in movie_words:
    for j in i:
        x.append(j)
for k in pos:
    for l in k:
        y.append(l) 


df = pd.DataFrame(list(zip(x, y)),  
               columns =['lemma', 'PoS'])
df = df[df["lemma"].str.len() >= 3]
df = df[df["PoS"].str.contains("n|a")]
df = df[["lemma"]].drop_duplicates()
df.to_csv("MovieWords.txt", index = False, header = False) 