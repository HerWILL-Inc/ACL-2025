from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer
import torch
import argparse
from bias_metrics import WordEmbeddingAssociationTest,EmbeddingCoherenceTest
from dataprocess import ProcessData
from bangla_fast_text_model import BanglaFasttext
import numpy as np
import math
import random
from tqdm import tqdm
import xlsxwriter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

skip_ptest=False
max_workers=10
modelnames = [
             "csebuetnlp/banglabert", 
             "saiful9379/Bangla_GPT2",
             "flax-community/gpt2-bengali",
             "ritog/bangla-gpt2",
             "csebuetnlp/banglat5",
            #  "banglafasttext",
             "neuropark/sahajBERT",
            #  "foysal-sentencebert", # will not work. Will work on this later
             "Kowsher/bangla-bert",
             "csebuetnlp/banglishbert", 
             "sagorsarker/bangla-bert-base", # this is equivalent to https://sparknlp.org/2022/04/11/bert_embeddings_bangla_bert_base_bn_3_0.html
             "shahidul034/text_generation_bangla_model"
            #  "sbnltk-sentence-transformer"
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
    #pre process
    #word = '[CLS]'+word+'[SEP]'
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
            with torch.no_grad():
                outputs=model.encoder(fake_inputs)
            embeddings = outputs["last_hidden_state"].mean(dim=1) # last layer from the encoder part
            return embeddings

        else:
            
            fake_inputs = tokenizer.encode(word, return_tensors="pt")
            with torch.no_grad():
                outputs = model(fake_inputs)
            
            
            #pred=outputs["logits"]
            hidden_states=outputs["hidden_states"]
            #attention=outputs["attentions"]
            #mean pooling
            embedding=hidden_states[-1] # last layer
            
            return embedding.mean(dim=1) # mean pooling
            
def get_X_Y_A_B_embeddings(model,modelname,tokenizer,X_male,Y_female,attr_1,attr_2):
    X_male=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in X_male])
    Y_female=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in Y_female])
    attr_1=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in attr_1])
    attr_2=torch.stack([get_embeddings(model,tokenizer,word,modelname) for word in attr_2])
    return X_male,Y_female,attr_1,attr_2

def get_embeddings_all_word(model,modelname,tokenizer,X_male,Y_female,attr_1,attr_2):
    cache={
        
    }
    words=X_male+Y_female+attr_1+attr_2
    print(f"Caching all word embeddings for this particular model.....")
    for word in tqdm(words):
        cache[word]=get_embeddings(model,tokenizer,word,modelname)
    return cache

def get_cached_embeddings(cache,X_hat,Y_hat,attr_1,attr_2):
    X_male_tensor=torch.stack([cache[word] for word in X_hat])
    Y_female_tensor=torch.stack([cache[word] for word in Y_hat])
    attr_1_tensor=torch.stack([cache[word] for word in attr_1])
    attr_2_tensor=torch.stack([cache[word] for word in attr_2])
    
    return X_male_tensor,Y_female_tensor,attr_1_tensor,attr_2_tensor
    
def p_test(
    model,
    modelname,
    tokenizer,
    male_words,
    female_words,
    male_attrs,
    female_attrs,
    weatfinder,
    n=100,


):
    X_Y_union=male_words+female_words
    runs=np.min((n,math.factorial(len(X_Y_union))))
    seen=set()
    X_male,Y_female,male_attr,female_attr=get_X_Y_A_B_embeddings(model,modelname,tokenizer,male_words,female_words,male_attrs,female_attrs)
    original=weatfinder(X_male,Y_female,male_attr,female_attr,test_stat=True).item()

    r=0
    print("Conducting p-test please be patient.")
    for i in tqdm(range(runs)):
        perm=tuple(random.sample(X_Y_union, len(X_Y_union)))
        if perm not in seen:
            X_hat = perm[0:len(male_words)]
            Y_hat = perm[len(female_words):]
            X_male,Y_female,male_attr,female_attr=get_X_Y_A_B_embeddings(model,modelname,tokenizer,X_hat,Y_hat,male_attrs,female_attrs)
            test_stat=weatfinder(X_male,Y_female,male_attr,female_attr,test_stat=True).item()
            if test_stat > original:
                r += 1
            seen.add(perm)
    p_value = r / runs
    return p_value





def p_test_multithreaded(
    model,
    modelname,
    tokenizer,
    male_words,
    female_words,
    male_attrs,
    female_attrs,
    weatfinder,
    n=100,
    num_threads=16
):
    X_Y_union = male_words + female_words
    runs = np.min((n, math.factorial(len(X_Y_union))))
    seen = set()
    X_male, Y_female, male_attr, female_attr = get_X_Y_A_B_embeddings(model, modelname, tokenizer, male_words, female_words, male_attrs, female_attrs)
    original = weatfinder(X_male, Y_female, male_attr, female_attr, test_stat=True).item()
    cache=get_embeddings_all_word(model,modelname,tokenizer,male_words,female_words,male_attrs,female_attrs)
    
    def permutation_test(perm):
        nonlocal seen, r,cache
        if perm not in seen:
            X_hat = perm[0:len(male_words)]
            Y_hat = perm[len(female_words):]
            #X_male, Y_female, male_attr, female_attr = get_X_Y_A_B_embeddings(model, modelname, tokenizer, X_hat, Y_hat, male_attrs, female_attrs)
            X_male, Y_female, male_attr, female_attr = get_cached_embeddings(cache,X_hat,Y_hat,male_attrs,female_attrs)
            target_embeddings1=X_male
            target_embeddings2=Y_female
            attribute_embeddings1=male_attr
            attribute_embeddings2=female_attr
            
            target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
            target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
            attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
            attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)

            # Normalize
            target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
            target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
            attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
            attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)

            # Compute cosine similarities
            X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
            X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
            Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
            Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
            X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
            X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])

            s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
            s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
            s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
            #S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        
            if s_X_Y_A_B > original:
                r += 1
            seen.add(perm)

    r = 0
    print("Conducting p-test please be patient.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(permutation_test, tuple(random.sample(X_Y_union, len(X_Y_union)))) for _ in range(runs)]
        for _ in tqdm(as_completed(futures), total=runs):
            pass

    p_value = r / runs
    return p_value

def calculate_weat_score_and_p_value(model,modelname,tokenizer,X,Y,A,B,weatfinder):
    global skip_ptest
    X_male,Y_female,attr_1,attr_2=get_X_Y_A_B_embeddings(model,modelname,tokenizer,X,Y,A,B)

    weat_score=weatfinder(
        target_embeddings1=X_male,
        target_embeddings2=Y_female,
        attribute_embeddings1=attr_1,
        attribute_embeddings2=attr_2

    )
   
    print(f"Weat score (cohens d actually): {weat_score.item()}")
    if not skip_ptest:
        p_value=p_test_multithreaded(model,modelname,tokenizer,X,Y,A,B,weatfinder)
        print(f"P value: {p_value}")
    else:
        p_value=-1 
    return weat_score.item(),p_value





def main():
    modelname=parse_arguments()
    model,tokenizer=initialize_model_and_tokenizer(modelname)
    weat=WordEmbeddingAssociationTest()
    #ect=EmbeddingCoherenceTest()
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

    workbook = xlsxwriter.Workbook('weat_socres.xlsx')
    worksheet = workbook.add_worksheet()

    # Define column names
    column_names = ['Model', 'WEAT score', 'p value']

    # Write column names to the first row
    for col_num, column_name in enumerate(column_names):
        worksheet.write(0, col_num, column_name)

    print(f"Selected attributes: {select_attrs['attr_1']} vs {select_attrs['attr_2']}")
    
    for row_num,modelname in enumerate(modelnames,start=1):

        model,tokenizer=initialize_model_and_tokenizer(modelname)

        w,p = calculate_weat_score_and_p_value(model,
                                     modelname,
                                     tokenizer,
                                     select_attrs["attr_1"],
                                     select_attrs["attr_2"],
                                     male_words,
                                     female_words,
                                     weat
                                    )
        data = [modelname, w, p]
        for col_num, cell_data in enumerate(data):
            worksheet.write(row_num,col_num,cell_data)
    
    workbook.close()
        

if __name__=='__main__':
    main()