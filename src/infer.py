import pickle
import os
import pandas as pd
import torch
import random
from tqdm import tqdm

import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename, folder = "./trained_model/"):
    completeName = os.path.join(folder, filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args, net, tokenizer):
        self.args = args
        self.net = net
        self.tokenizer = tokenizer

        self.cuda = torch.cuda.is_available()
        
        logger.info("Restoring model...")
        from .train_funcs import load_state
        if self.cuda:
            self.net.cuda()
        load_state(self.net, None, None, self.args, load_best=True)
        logger.info("Done!")
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl", args.model_save_path)

    def get_e1e2_start(self, x):
        e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
                        [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start
    
    def infer_sentence(self, sentence):
        self.net.eval()
        replace_vocab = self.tokenizer.convert_ids_to_tokens(random.sample(range(2000, self.tokenizer.vocab_size-2000), 2))
        sentence = sentence.replace("@GENE$","[E1]%s[/E1]" % replace_vocab[0])
        sentence = sentence.replace("@DISEASE$","[E2]%s[/E2]" % replace_vocab[1])
        tokenized = self.tokenizer.encode(sentence); #print(tokenized)
        e1_e2_start = self.get_e1e2_start(tokenized); #print(e1_e2_start)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
        attention_mask = (tokenized != self.pad_id).float()
        token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()
        
        if self.cuda:
            tokenized = tokenized.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        with torch.no_grad():
            classification_logits = self.net(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                                        e1_e2_start=e1_e2_start)
            predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()
        #print("Sentence: ", sentence)
        print("Predicted: ", bool(self.rm.idx2rel[predicted]), '\n')
        return bool(self.rm.idx2rel[predicted])