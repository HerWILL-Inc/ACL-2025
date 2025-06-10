from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer
import torch
import argparse
from bias_metrics import WordEmbeddingAssociationTest,EmbeddingCoherenceTest, CosMetric
from dataprocess import ProcessData
from bangla_fast_text_model import BanglaFasttext
import numpy as np
import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
# from sbnltk_stransformer import SentenceEncoder
torch.set_grad_enabled(False)

modelnames = [
            #  "banglafasttext",
             "csebuetnlp/banglabert", 
             "saiful9379/Bangla_GPT2",
             "flax-community/gpt2-bengali",
             "ritog/bangla-gpt2",
             "csebuetnlp/banglat5",
             "neuropark/sahajBERT",
             "Kowsher/bangla-bert",
             "csebuetnlp/banglishbert", 
             "sagorsarker/bangla-bert-base", # this is equivalent to https://sparknlp.org/2022/04/11/bert_embeddings_bangla_bert_base_bn_3_0.html
             "shahidul034/text_generation_bangla_model"
            #  "sbnltk-sentence-transformer"
] 

x_correlations = []
y_correlations = []


def initialize_model_and_tokenizer(modelname):
    if modelname not in modelnames:
        raise Exception("This is not a supported model name")
    if modelname=="banglafasttext":
        model = BanglaFasttext(model_path = 'C:/Users/NMK/AppData/Local/Programs/Python/Python39/Lib/site-packages/gensim/test/test_data/cache/models/fasttext/Bangla_fasttext_skipgram.pickle')
        model.model_load()
        return model,None
    elif modelname=="foysal-sentencebert":
        """
        Please run the download_st.py first this will download this model in a folder named ./Towhid-Sus....
        """
        model=SentenceTransformer("./Towhid-Sust-transformer")
        return model,None
    elif modelname=="sbnltk-sentence-transformer":
        model=SentenceEncoder()
        return model,None

    else:
        from_tf=False
        if modelname=="saiful9379/Bangla_GPT2" or modelname=="shahidul034/text_generation_bangla_model":
            from_tf=True
        model = AutoModelForPreTraining.from_pretrained(
            modelname,
            output_hidden_states=True,
            output_attentions=True,
            cache_dir="cache/models/transformers",
            from_tf=from_tf
        )
        tokenizer = AutoTokenizer.from_pretrained(
            modelname,
            cache_dir="cache/models/transformers"
        )
        print(model)
        return model,tokenizer

def get_embeddings(model,tokenizer,word,modelname):
    if modelname=="banglafasttext":
        embedding=torch.from_numpy(model.sent_embd(word)) # this is prone to problem uses huge amount of ram
        return embedding
    if modelname=="foysal-sentencebert":
        embeddings=model.encode(word)
        return embeddings # word will break as it is sentence transformer returns {key:value}
    if modelname=="sbnltk-sentence-transformer":
        embeddings=model.convert_text_to_embeddings(word)
        return embeddings
    else:
        word=normalize(word) 
        #fake_tokens = tokenizer.tokenize(word)
        if 't5' in modelname:
            fake_inputs = tokenizer.encode(word, return_tensors="pt")
            outputs=model.encoder(fake_inputs)
            embeddings = outputs["last_hidden_state"].mean(dim=1) # last layer from the encoder part
            return embeddings

        else:
            fake_inputs = tokenizer.encode(word, return_tensors="pt")
            outputs = model(fake_inputs)

            
            #pred=outputs["logits"]
            hidden_states=outputs["hidden_states"]
            #attention=outputs["attentions"]
            embedding=hidden_states[-1] # last layer
            return embedding.mean(dim=1) # mean pooling


def get_X_Y_A_embeddings(model,modelname,tokenizer,X_male,Y_female,attr):
    X_male=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in X_male])
    Y_female=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in Y_female])
    attr=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in attr])
    return X_male,Y_female,attr

def calculatebiasofmodel(model,modelname,tokenizer,X,Y,A,cosmetric):
    X_male,Y_female,attr=get_X_Y_A_embeddings(model,modelname,tokenizer,X,Y,A)

    x,y = cosmetric(
        target_embeddings1=X_male,
        target_embeddings2=Y_female,
        attribute_embeddings=attr,
    )

    x_correlations.append(x)
    y_correlations.append(y)

def main():


    for modelname in modelnames:
        model,tokenizer=initialize_model_and_tokenizer(modelname)
        weat=WordEmbeddingAssociationTest()
        cos = CosMetric()
        dataloader=ProcessData()


        male_words=dataloader.get_attributes("male_terms")
        female_words=dataloader.get_attributes("female_terms")
        # stem_words=dataloader.get_attributes("stem_words")
        shape_words = dataloader.get_attributes('shape_words')
        
        # calculatebiasofmodel(model,modelname,tokenizer,male_words,female_words,stem_words,cos)
        calculatebiasofmodel(model,modelname,tokenizer,male_words,female_words,shape_words,cos)

    x = [float(c) for c in x_correlations]
    y = [float(c) for c in y_correlations]

    print(x)
    print(y)
    print(modelnames)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_correlations,
        y=y_correlations,
        mode="markers+text",
        text=modelnames,
        textposition="top center"
    ))

    fig.show()

if __name__=='__main__':
    main()