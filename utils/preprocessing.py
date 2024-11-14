import os
import re
import csv
import json
import nltk
import codecs
import random
import unicodedata
import itertools
import numpy as np

import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

import config

word_pairs = {"it's":"it is", 
              "don't":"do not", 
              "doesn't":"does not", 
              "didn't":"did not", 
              "you'd":"you would", 
              "you're":"you are", 
              "you'll":"you will", 
              "i'm":"i am", 
              "they're":"they are", 
              "that's":"that is", 
              "what's":"what is", 
              "couldn't":"could not", 
              "i've":"i have", 
              "we've":"we have", 
              "can't":"cannot", 
              "i'd":"i would", 
              "i'd":"i would", 
              "aren't":"are not", 
              "isn't":"is not", 
              "wasn't":"was not", 
              "weren't":"were not", 
              "won't":"will not", 
              "there's":"there is", 
              "there're":"there are", 
              "let's":"let us"}


#print lines cornell corpus
def printLines(file, n=10):
    corpus_name = "movie-corpus"
    corpus = os.path.join(dir_corpus, corpus_name)
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#printLines(os.path.join(corpus, "utterances.jsonl"))

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def create_formated_cornell_corpus():
    # Define path to new file
    corpus = os.path.join(config.dir_corpus, config.corpus_name)
    datafile = os.path.join("data", config.formated_cornell)

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    mocie_lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
    movie_conversations_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), mocie_lines_fields)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, movie_conversations_fields)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.strip())
    s = re.sub(r"([.!?,'])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,']+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def clean(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    return normalizeString(sentence)

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[clean(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < config.MAX_LENGTH and len(p[1].split(' ')) < config.MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(config.MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = {'idx': [], 'context': [], 'target': [], 'target_gen': [], 'input_dis': []}

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in nltk.word_tokenize(input_sentence):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in nltk.word_tokenize(output_sentence):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs['context'].append(pair[0])
            keep_pairs['target'].append(pair[1])
            keep_pairs['target_gen'].append('')
            keep_pairs['input_dis'].append('')

    if config.max_pairs is not None:
        keep_pairs_max = {'idx': [], 'context': [], 'target': [], 'target_gen': [], 'input_dis': []}
        keep_pairs_max['context'] = keep_pairs['context'][0:config.max_pairs]
        keep_pairs_max['target'] = keep_pairs['target'][0:config.max_pairs]
        keep_pairs_max['target_gen'] = keep_pairs['target_gen'][0:config.max_pairs]
        keep_pairs_max['input_dis'] = keep_pairs['input_dis'][0:config.max_pairs]
        keep_pairs = keep_pairs_max

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs['target']), len(keep_pairs['target']) / len(pairs)))
    
    return keep_pairs


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_token: "PAD", config.SOS_token: "SOS", config.EOS_token: "EOS", config.UNK_token: "UNK"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def __len__(self):
        return self.num_words

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_token: "PAD", config.SOS_token: "SOS", config.EOS_token: "EOS", config.UNK_token: "UNK"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)



def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long() ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    def binary_matrix(l, value=config.PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == config.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info['context'])
    input_dis, input_dis_lengths = merge(item_info['input_dis'])

    ## Target
    target_gen_batch, target_gen_lengths = merge(item_info['target_gen'])
    target_batch, target_lengths = merge(item_info['target'])
    '''print('context: ', item_info['context'])
    print('input_batch: ', input_batch)
    print('input_lengths: ', input_lengths)
    print('input_lengths: ', torch.LongTensor(input_lengths))
    print('----------------------')
    print('target_text: ', item_info['target'])
    print('target_batch: ', target_batch)
    print('target_lengths: ', target_lengths)
    print('target_lengths: ', target_lengths)
    print('----------------------')'''


    d = {}
    d["idx"] = item_info['idx']
    #input sentence
    d["input_batch"] = input_batch
    d["input_batch_length"] = torch.LongTensor(input_lengths)

    #senteces created by generator
    d["target_gen_batch"] = target_gen_batch
    d["target_gen_batch_length"] = torch.LongTensor(target_gen_lengths)

    #real response sentences
    d["target_batch"] = target_batch
    d["mask"] = torch.ByteTensor(binary_matrix(target_batch))
    d["target_batch_length"] = torch.LongTensor(target_lengths)

    #target discriminator(0-fake, -real)
    d["target_dis"] = item_info['target_dis']

    #input text discriminator based in the target_dis
    d["input_dis_batch"] = input_dis
    d["input_dis_batch_length"] = torch.LongTensor(input_dis_lengths)

    
    return d


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data 
        self.set_idx_data()
        self.set_target_dis()

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["idx"] = self.data["idx"][index]
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["target_gen_text"] = self.data["target_gen"][index]
        item["input_dis_text"] = self.data["input_dis"][index]
        item["target_dis"] = self.data["target_dis"][index]

        item["context"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_gen"] = self.preprocess(item["target_gen_text"], anw=True)
        item["input_dis"] = self.preprocess(item["input_dis_text"])

        return item

    def update_value(self, gen, dis):
        self.data['target_gen'] = gen
        self.data['input_dis'] = dis


    def preprocess(self, sentence, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] 
                    if word in self.vocab.word2index 
                    else config.UNK_token for word in nltk.word_tokenize(sentence)] + [config.EOS_token]

            return torch.LongTensor(sequence)
        else:
            seq_in = [config.SOS_token]
            seq_in += [self.vocab.word2index[word] if word in self.vocab.word2index 
                        else config.UNK_token for word in nltk.word_tokenize(sentence)]

            return torch.LongTensor(seq_in)

    def set_idx_data(self):
        arr =[i for i in range(self.__len__())]
        self.data['idx'] = arr

    def set_target_dis(self):
        size_ = len(self.data['context'])
        arr = np.zeros(size_, dtype=int)
        arr[:int(size_/2)] = 1
        np.random.seed(42) 
        np.random.shuffle(arr)
        arr = arr.tolist()
        self.data['target_dis'] = arr

def create_dataset(input_, target_, target_gen, voc):
    return Dataset(
        {'context': input_, 
        'target': target_, 
        'target_gen': target_gen, 
        'input_dis': target_gen}, voc)

def get_div_dataset(pairs, voc):
    input_train, input_aux, target_train, target_aux, target_gen_train, target_gen_aux = train_test_split(
                pairs['context'], pairs['target'], pairs['target_gen'], test_size=0.2, random_state=0)
    
    input_pre, input_test, target_pre, target_test, target_gen_pre, target_gen_test = train_test_split(
                input_aux, target_aux, target_gen_aux, test_size=0.5, random_state=0)
    
    dataset_train = create_dataset(input_train, target_train, target_gen_train, voc)
    dataset_pre = create_dataset(input_pre, target_pre, target_gen_pre, voc)
    dataset_test = create_dataset(input_test, target_test, target_gen_test, voc)

    return dataset_train, dataset_pre, dataset_test


def load_data():
    if not os.path.exists("data/formatted_movie_lines.txt"):
        create_formated_cornell_corpus()

    datafile = os.path.join("data", config.formated_cornell)

    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(config.corpus_name, datafile, save_dir)
    pairs = trimRareWords(voc, pairs)
    '''
    for idx in range(10):
        print(pairs['context'][idx])
        print(pairs['target'][idx])
        print('------------------------------')
    '''
    
    train_dataset, pre_dataset, test_dataset = get_div_dataset(pairs, voc)


    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    data_loader_pre = torch.utils.data.DataLoader(dataset=pre_dataset, batch_size=config.batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    return data_loader_train, data_loader_pre, data_loader_test, voc



