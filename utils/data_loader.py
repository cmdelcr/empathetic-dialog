import torch
import torch.utils.data as data
import random
import math
import os
import logging 
import pandas as pd
from copy import deepcopy
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from model.common import write_config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


word_pairs = {"it's":"it is", "don't":"do not", 
              "he's":"he is", "she's":"she is",
              "doesn't":"does not", "didn't":"did not", 
              "you'd":"you would", "you're":"you are", 
              "you'll":"you will", "i'm":"i am", 
              "they're":"they are", "that's":"that is", 
              "what's":"what is", "couldn't":"could not", 
              "i've":"i have", "we've":"we have", 
              "can't":"cannot", "i'd":"i would", 
              "i'd":"i would", "aren't":"are not", 
              "isn't":"is not", "wasn't":"was not", 
              "weren't":"were not", "won't":"will not", 
              "there's":"there is", "there're":"there are"}
              
word_pairs_extended = {"n't": " not", "'re":" are", "'d":" would", "'ll":" will", "'ve": " have"}

emotion_lexicon = pickle.load(open('dataset/empathetic_dialogs/vad_vocab.pkl', 'rb'))

dict_emo = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 
            'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 
            'hopeful': 13, 'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 
            'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 
            'apprehensive': 30, 'faithful': 31}


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v).replace('\\', '')
    for k, v in word_pairs_extended.items():
        sentence = sentence.replace(k,v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

def get_vad_values(target, vocab_emo=None, type_='dataset'):
    e_list = []
    target = clean(target, word_pairs) if type_ != 'dataset' else target
    ws_pos = nltk.pos_tag(target)  # pos
    for w in ws_pos:
        if w[0] not in stop_words and (w[1].startswith('J') or w[0] in emotion_lexicon):
            e_list.append(w[0])
            if type_ == 'dataset':
                vocab_emo.index_word(w[0])

    return (e_list, vocab_emo) if type_ == 'dataset' else e_list

def read_langs(vocab):
    vocab_emo = deepcopy(vocab)
    train_context = np.load('dataset/empathetic_dialogs/sys_dialog_texts.train.npy',allow_pickle=True)
    train_target = np.load('dataset/empathetic_dialogs/sys_target_texts.train.npy',allow_pickle=True)
    train_emotion = np.load('dataset/empathetic_dialogs/sys_emotion_texts.train.npy',allow_pickle=True)
    train_situation = np.load('dataset/empathetic_dialogs/sys_situation_texts.train.npy',allow_pickle=True)

    dev_context = np.load('dataset/empathetic_dialogs/sys_dialog_texts.dev.npy',allow_pickle=True)
    dev_target = np.load('dataset/empathetic_dialogs/sys_target_texts.dev.npy',allow_pickle=True)
    dev_emotion = np.load('dataset/empathetic_dialogs/sys_emotion_texts.dev.npy',allow_pickle=True)
    dev_situation = np.load('dataset/empathetic_dialogs/sys_situation_texts.dev.npy',allow_pickle=True)
    
    test_context = np.load('dataset/empathetic_dialogs/sys_dialog_texts.test.npy',allow_pickle=True)
    test_target = np.load('dataset/empathetic_dialogs/sys_target_texts.test.npy',allow_pickle=True)
    test_emotion = np.load('dataset/empathetic_dialogs/sys_emotion_texts.test.npy',allow_pickle=True)
    test_situation = np.load('dataset/empathetic_dialogs/sys_situation_texts.test.npy',allow_pickle=True)

    data_train = {'context':[],'emotion_context':[],'target':[],'target_emo':[],'emotion':[], 'situation':[]}
    data_dev = {'context':[],'emotion_context':[],'target':[],'target_emo':[],'emotion':[], 'situation':[]}
    data_test = {'context':[],'emotion_context':[],'target':[],'target_emo':[],'emotion':[], 'situation':[]}

    for context in train_context:
        u_list, e_list = [], []
        for u in context:
            u = clean(u, word_pairs)
            ws_pos = nltk.pos_tag(u)  # pos
            for w in ws_pos:
                if w[0] not in stop_words and (w[1].startswith('J') or w[0] in emotion_lexicon):
                    e_list.append(w[0])
                    vocab_emo.index_word(w[0])
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(u_list)
        data_train['emotion_context'].append(e_list)
    
    for target in train_target:
        target = clean(target, word_pairs)
        e_list, vocab_emo = get_vad_values(target, vocab_emo=vocab_emo)
        data_train['target'].append(target)
        data_train['target_emo'].append(e_list)
        vocab.index_words(target)

    for situation in train_situation:
        situation = clean(situation, word_pairs)
        data_train['situation'].append(situation)
        vocab.index_words(situation)

    for emotion in train_emotion:
        data_train['emotion'].append(emotion)

    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['situation'])


    for context in dev_context:
        u_list, e_list = [], []
        for u in context:
            u = clean(u, word_pairs)
            ws_pos = nltk.pos_tag(u)  # pos
            for w in ws_pos:
                if w[0] not in stop_words and (w[1].startswith('J') or w[0] in emotion_lexicon):
                    e_list.append(w[0])
                    vocab_emo.index_word(w[0])
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(u_list)
        data_dev['emotion_context'].append(e_list)

    for target in dev_target:
        target = clean(target, word_pairs)
        e_list, vocab_emo = get_vad_values(target, vocab_emo=vocab_emo)
        data_dev['target'].append(target)
        data_dev['target_emo'].append(e_list)
        vocab.index_words(target)

    for situation in dev_situation:
        situation = clean(situation, word_pairs)
        data_dev['situation'].append(situation)
        vocab.index_words(situation)

    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)

    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['situation'])


    for context in test_context:
        u_list, e_list = [], []
        for u in context:
            u = clean(u, word_pairs)
            ws_pos = nltk.pos_tag(u)  # pos
            for w in ws_pos:
                if w[0] not in stop_words and (w[1].startswith('J') or w[0] in emotion_lexicon):
                    e_list.append(w[0])
                    vocab_emo.index_word(w[0])
            u_list.append(u)
            vocab.index_words(u)
        data_test['context'].append(u_list)
        data_test['emotion_context'].append(e_list)
    for target in test_target:
        target = clean(target, word_pairs)
        e_list, vocab_emo = get_vad_values(target, vocab_emo=vocab_emo)
        data_test['target'].append(target)
        data_test['target_emo'].append(e_list)
        vocab.index_words(target)

    for situation in test_situation:
        situation = clean(situation, word_pairs)
        data_test['situation'].append(situation)
        vocab.index_words(situation)

    for emotion in test_emotion:
        data_test['emotion'].append(emotion)

    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])

    return data_train, data_dev, data_test, vocab, vocab_emo


def load_dataset():
    if(os.path.exists('dataset/empathetic_dialogs/dataset_preproc.p')):
        print("LOADING empathetic_dialogue")
        with open('dataset/empathetic_dialogs/dataset_preproc.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab, vocab_emo] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab, vocab_emo = read_langs(vocab=Lang(
                {config.UNK_idx: "UNK", config.PAD_idx: "PAD", 
                 config.EOS_idx: "EOS", config.SOS_idx: "SOS", 
                 config.USR_idx:"USR", config.SYS_idx:"SYS", 
                 config.CLS_idx:"CLS", config.EMO_idx:"EMO",
                 config.SEP_idx:"SEP"})) 
        
        with open('dataset/empathetic_dialogs/dataset_preproc.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab, vocab_emo], f)
            print("Saved PICKLE")

    return data_tra, data_val, data_tst, vocab, vocab_emo


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab, vocab_emo):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.vocab_emo = vocab_emo
        self.data = data 
        self.emo2index = dict_emo
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context_text"] = self.data["emotion_context"][index]
        item["target_emo_text"] = self.data["target_emo"][index]


        inputs = self.preprocess([item["context_text"], item["emotion_context_text"]])
        item["context"], item["context_ext"], item["oovs"], item["context_mask"], \
        item["emotion_context"], item["emotion_context_ext"], item["emotion_context_mask"] = inputs

        item["target_emo"], item["target_emo_mask"] = self.preprocess_te(item["target_emo_text"], emo=True)
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo2index)
        
        return item


    def target_oovs(self, target, oovs):  #
        ids = []
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(config.UNK_idx)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(config.EOS_idx)
        return torch.LongTensor(ids)


    def process_oov(self, context, emotion_context):  # oov for input
        ids = []
        ids_e = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.vocab.word2index) + oov_num)

        for ew in emotion_context:
            if ew in self.vocab.word2index:
                i = self.vocab.word2index[ew]
                ids_e.append(i)
            elif ew in oovs:
                oov_num = oovs.index(ew)
                ids_e.append(len(self.vocab.word2index) + oov_num)
            else:
                oovs.append(ew)
                oov_num = oovs.index(w)
                ids_e.append(len(self.vocab.word2index) + oov_num)
        return ids, ids_e, oovs


    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else \
                        config.UNK_idx for word in arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:

            context = arr[0]
            emotion_context = arr[1]

            X_dial = [config.CLS_idx]
            X_dial_ext = [config.CLS_idx]
            X_dial_mask = [config.CLS_idx]

            X_emotion = [config.EMO_idx]
            X_emotion_ext = [config.EMO_idx]
            X_emotion_mask = [config.EMO_idx]

            for i, sentence in enumerate(context):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_dial_mask += [spk for _ in range(len(sentence))] 

            for i, ew in enumerate(emotion_context):
                X_emotion += [self.vocab_emo.word2index[ew] if ew in self.vocab_emo.word2index else config.UNK_idx]
                X_emotion_mask += [self.vocab_emo.word2index["EMO"]]

            X_ext, X_e_ext, X_oovs = self.process_oov(context, emotion_context)
            X_dial_ext += X_ext
            X_emotion_ext += X_e_ext

            assert len(X_dial) == len(X_dial_mask) == len(X_dial_ext)
            assert len(X_emotion) == len(X_emotion_ext) == len(X_emotion_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_dial_ext), torch.LongTensor(X_oovs), \
                    torch.LongTensor(X_dial_mask), torch.LongTensor(X_emotion), torch.LongTensor(X_emotion_ext), \
                    torch.LongTensor(X_emotion_mask)

    def preprocess_te(self, arr, anw=False, emo=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else \
                        config.UNK_idx for word in arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            X_dial = [config.CLS_idx if not emo else config.EMO_idx]
            X_mask = [config.CLS_idx if not emo else config.EMO_idx]

            if emo:
                for i, ew in enumerate(arr):
                    X_dial += [self.vocab_emo.word2index[ew] if ew in self.vocab_emo.word2index else config.UNK_idx]
                    X_mask += [self.vocab_emo.word2index["EMO"]]
                    #print(X_mask)
            else:
                for i, sentence in enumerate(arr):
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                    spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                    X_mask += [spk for _ in range(len(sentence))] 
            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)


    def preprocess_emo(self, emotion, emo2index):
        program = [0]*len(emo2index)
        program[emo2index[emotion]] = 1
        return program, emo2index[emotion]

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    
    ## input
    input_batch, input_lengths     = merge(item_info['context'])
    context_ext_batch, _ = merge(item_info['context_ext'])
    mask_input, mask_input_lengths = merge(item_info['context_mask'])

    ## input - emotion_context
    emotion_context_batch, emotion_context_lengths = merge(item_info['emotion_context'])
    emotion_context_ext_batch, _ = merge(item_info['emotion_context_ext'])
    mask_emotion_context, _ = merge(item_info['emotion_context_mask'])

    ## Target
    target_batch, target_lengths   = merge(item_info['target'])
    target_ext_batch, _ = merge(item_info['target_ext'])
    target_emo_batch, target_emo_lengths = merge(item_info['target_emo'])
    mask_target_emo, _ = merge(item_info['target_emo_mask'])


    d = {}
    d["input_batch"] = input_batch.to(config.device)
    d["input_ext_batch"] = context_ext_batch.to(config.device)  # (bsz, max_context_len)
    d[" "] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input.to(config.device)


    d["emotion_context"] = emotion_context_batch.to(config.device)
    d["emotion_context_ext"] = emotion_context_ext_batch.to(config.device)  # (bsz, max_emo_context_len)
    d["emotion_context_lengths"] = torch.LongTensor(emotion_context_lengths)
    d["mask_emotion_context"] = mask_emotion_context.to(config.device)

    d["target_emo"] = target_emo_batch.to(config.device)
    d["target_emo_lengths"] = torch.LongTensor(target_emo_lengths)
    d["mask_target_emo"] = mask_target_emo.to(config.device)

    d["target_batch"] = target_batch.to(config.device)
    d["target_ext_batch"] = target_ext_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']
    d["emotion_label"] = item_info['emotion_label']

    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']
    d["emotion_context_text"] = item_info['emotion_context_text']
    d["target_emo_text"] = item_info['target_emo_text']
    d["oovs"] = item_info["oovs"]

    # d["target_emotion_scores"] = item_info["target_emotion_scores"]
    
    return d


def prepare_data_seq(batch_size=32, write_config_val=True):  

    pairs_tra, pairs_val, pairs_tst, vocab, vocab_emo = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab, vocab_emo)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab, vocab_emo)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    dataset_test = Dataset(pairs_tst, vocab, vocab_emo)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)
    if write_config_val:
        write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, vocab_emo, dict_emo
