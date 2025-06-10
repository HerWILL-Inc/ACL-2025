import gdown
import zipfile
import pickle
import numpy as np
import os
import logging
from gensim.models.fasttext import FastText
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import fasttext
from gensim.test.utils import datapath
import torch


class BanglaFasttext:
    def __init__(self, model_path=None, method='cbow', save_path = ''):
            try:
                try:
                    model = FastText.load(model_path)
                except:
                    with open(model_path, 'rb') as handle:
                        model = pickle.load(handle)  
                self.modelwv =model.wv
                self.model=model   
            except:
                self.modelwv = fasttext.load_facebook_vectors(datapath(model_path))           

    def model_load(self):
        return self.modelwv

        
    def div_norm(self, x):
        
        norm_value = np.sqrt(np.sum(x**2))

        if norm_value > 0:
            return x * ( 1.0 / norm_value)
        else:
            return x


    def sent_embd(self,corpus):
        emd_corpus=[]
        for sent in tqdm(corpus):
            try:
                embd = [self.div_norm(self.modelwv[i]) for i in sent.split()]
                emd_corpus.append(list(np.mean(embd, axis=0)))
            except:
                emd_corpus.append(self.div_norm(self.modelwv['unknown']))
        return np.asarray(emd_corpus)


    def word_similarity(self, word1, word2):
        return cosine_similarity(self.modelwv[word1].reshape(1, -1), self.modelwv[word2].reshape(1, -1))[0]
    def sent_similarity(self, sent1, sent2):
        sent1=self.sent_embd(sent1)
        sent2=self.sent_embd(sent2)

        return cosine_similarity(self.modelwv[sent1].reshape(1, -1), self.modelwv[sent2].reshape(1, -1))[0]


    def fine_tuning(self,corpus, epochs=5):
        sent=[]
        for k in corpus:
            sent.append(k.split())
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.model.build_vocab(corpus_iterable=sent, update=True)
        self.model.train(corpus_iterable=sent, total_examples = len(sent), epochs=epochs)
        self.modelwv = self.model.wv
        return self.model


