from sbnltk.Sentence_embedding import Bangla_sentence_embedding_hd
import numpy as np
import torch

class SentenceEncoder:
     def __init__(self, device):
         self.s2s = Bangla_sentence_embedding_hd()
         self.device = device

     def convert_text_to_embeddings(self, batch_text):
         stack = []
         print('batch text:',batch_text)
         for sent in batch_text:
            print('sent: ',sent)
            sent = [sent]
            sentence_embeddings = self.s2s.encode_sentence_list(sent)
            ea = np.array([])
            for key, value in sentence_embeddings.items():
                ea = np.hstack((ea,value))
            sentence_emb = torch.FloatTensor(ea).to(self.device).reshape(1,-1)
            stack.append(sentence_emb)
         output = torch.cat(stack, dim=0)
         return output