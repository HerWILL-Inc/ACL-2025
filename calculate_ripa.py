from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer
import torch
import argparse
from bias_metrics import RIPA
from dataprocess import ProcessData
from bangla_fast_text_model import BanglaFasttext
import numpy as np
import math
import random
from tqdm import tqdm
import xlsxwriter
torch.set_grad_enabled(False)

modelnames = [
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
] 
def parse_arguments():
    parser=argparse.ArgumentParser(
        description="Process Arguments."
    )

    help_text=f"Pass model name.List of available models are {','.join(modelnames)}"
    parser.add_argument(
        '--modelname',
        help=help_text,
        default="csebuetnlp/banglabert"
    )
    parsed=parser.parse_args()
    return parsed.modelname


def initialize_model_and_tokenizer(modelname):
    if modelname not in modelnames:
        raise Exception("This is not a supported model name")
    if modelname=="banglafasttext":
        model = BanglaFasttext(model_path = r'cache\models\fasttext\Bangla_fasttext_skipgram.pickle')
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

def get_X_Y_A_embeddings(model,modelname,tokenizer,X_male,Y_female,attr_1):
    X_male=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in X_male])
    Y_female=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in Y_female])
    attr_1=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in attr_1])
    
    return X_male,Y_female,attr_1


def calculate_RND_score(model,modelname,tokenizer,X,Y,A,metric):
    X_male,Y_female,attr_1=get_X_Y_A_embeddings(model,modelname,tokenizer,X,Y,A)

    # print('shape of x:'+str(X_male.shape))
    # print('shape of y:'+str(Y_female.shape))
    # print('shape of a:'+str(attr_1.shape))
    

    score=metric(
        target_embeddings1=X_male,
        target_embeddings2=Y_female,
        attribute_embeddings=attr_1,
    )
   
    print(f"RND score for model {modelname}: {score}")

    return score

def main():
    modelname=parse_arguments()
    model,tokenizer=initialize_model_and_tokenizer(modelname)
    ripa=RIPA()
    dataloader=ProcessData()


    male_words=dataloader.get_attributes("male_terms")
    female_words=dataloader.get_attributes("female_terms")
    stem_words=dataloader.get_attributes("stem_words")
    shape_words=dataloader.get_attributes("shape_words")
    
    """ 
    change  this to desired attributes
    """
    select_attrs={
        "attr_1":stem_words,
        "attr_2":shape_words
    }

    workbook = xlsxwriter.Workbook('RIPA_socres.xlsx')
    worksheet = workbook.add_worksheet()

    # Define column names
    column_names = ['Model', 'STEM RIPA Score', 'SHAPE RIPA Score']

    # Write column names to the first row
    for col_num, column_name in enumerate(column_names):
        worksheet.write(0, col_num, column_name)

    print(f"Selected attributes: {select_attrs['attr_1']} vs {select_attrs['attr_2']}")
    
    for row_num,modelname in enumerate(modelnames,start=1):

        model,tokenizer=initialize_model_and_tokenizer(modelname)

        stem_score = calculate_RND_score(model,
                                     modelname,
                                     tokenizer,
                                     male_words,
                                     female_words,
                                     select_attrs["attr_1"],
                                     ripa
                                    )
        
        shape_score = calculate_RND_score(model,
                                     modelname,
                                     tokenizer,
                                     male_words,
                                     female_words,
                                     select_attrs["attr_2"],
                                     ripa
                                    )
        

        data = [modelname, stem_score, shape_score]
        for col_num, cell_data in enumerate(data):
            worksheet.write(row_num,col_num,cell_data)
    
    workbook.close()
        

if __name__=='__main__':
    main()