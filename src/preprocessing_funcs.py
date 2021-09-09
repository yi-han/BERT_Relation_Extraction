import os
import re
import random
import copy
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

class RE_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
                                                             axis=1)
        
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                try:
                    self.rel2idx[relation] = int(relation)
                except ValueError:
                    self.rel2idx[relation] = self.n_classes
                    self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

def read_data(texts, replace_vocab, mode='train'):
    sents, relations = [], []

    for index, line in enumerate(texts):
        line = line.split("\t")
        sent = line[-2].strip()
        relation = line[-1].strip()
        sent = sent.replace("@GENE$","[E1]%s[/E1]" % replace_vocab[2*index])
        sent = sent.replace("@DISEASE$","[E2]%s[/E2]" % replace_vocab[2*index+1])
        #sent = sent.replace("@GENE$","[E1]GENE[/E1]")
        #sent = sent.replace("@DISEASE$","[E2]DISEASE[/E2]")
        sents.append(sent)
        relations.append(relation)
    return sents, relations

def preprocess_oscer(args, tokenizer):
    data_train_path = args.train_data
    data_test_path = args.test_data

    with open(data_train_path, 'r', encoding='utf8') as f:
        texts_train = f.readlines()

    with open(data_test_path, 'r', encoding='utf8') as f:
        texts_test = f.readlines()
        texts_test = texts_test[1:]

    replace_vocab = tokenizer.convert_ids_to_tokens(random.sample(range(2000, tokenizer.vocab_size-2000), \
                                  2*(len(texts_train) + len(texts_test))))

    logger.info("Reading training file %s..." % data_train_path)
    sents, relations = read_data(texts_train, replace_vocab[:2*len(texts_train)], 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    logger.info("Reading test file %s..." % data_test_path)
    sents, relations = read_data(texts_test, replace_vocab[2*len(texts_train):], 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('relations.pkl', rm, args.model_save_path)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('df_train.pkl', df_train, args.model_save_path)
    save_as_pickle('df_test.pkl', df_test, args.model_save_path)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

def load_dataset(args, tokenizer):
    relations_path = os.path.join(args.model_save_path, 'relations.pkl')
    train_path = os.path.join(args.model_save_path, 'df_train.pkl')
    test_path = os.path.join(args.model_save_path, 'df_test.pkl')
    if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
        rm = load_pickle('relations.pkl', args.model_save_path)
        df_train = load_pickle('df_train.pkl', args.model_save_path)
        df_test = load_pickle('df_test.pkl', args.model_save_path)
        logger.info("Loaded preproccessed data.")
    else:
        df_train, df_test, rm = preprocess_oscer(args, tokenizer)

    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    test_set = RE_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    train_set = RE_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    
    return train_set, test_set

def load_pretrained_model(args):
    if args.model_no == 0:
        from .model.BERT.modeling_bert import BertModel as Model
        from .model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model = args.model_size#'bert-large-uncased' 'bert-base-uncased'
        lower_case = True
        model_name = 'BERT'
        net = Model.from_pretrained(model, force_download=False, \
                                model_size=args.model_size,
                                task='classification',\
                                n_classes_=args.num_classes)
    elif args.model_no == 1:
        from .model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
        from .model.ALBERT.modeling_albert import AlbertModel as Model
        model = args.model_size #'albert-base-v2'
        lower_case = True
        model_name = 'ALBERT'
        net = Model.from_pretrained(model, force_download=False, \
                                model_size=args.model_size,
                                task='classification',\
                                n_classes_=args.num_classes)
        
    if os.path.isfile(os.path.join(args.model_save_path, "%s_tokenizer.pkl" % model_name)):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name, args.model_save_path)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer, args.model_save_path)
        logger.info("Saved %s tokenizer at %s%s_tokenizer.pkl" %(model_name, args.model_save_path, model_name))
    
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1

    net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer