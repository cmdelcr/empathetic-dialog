import os
import torch
import logging 
import argparse

# Default word tokens
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6
EMO_idx = 7 # emotional state
SEP_idx = 8 




# Arguemnts
parser = argparse.ArgumentParser(description='SeqGAN')

parser.add_argument("--dataset", type=str, default="empathetic")
parser.add_argument("--data_dir", type=str, default="data/empathetic_dialogs")
parser.add_argument("--vae", type=bool, default=False)
parser.add_argument("--eq6_loss", type=bool, default=False)
parser.add_argument("--vader_loss", type=bool, default=False)  # add vader loss
parser.add_argument("--init_emo_emb", action="store_true")

parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--decoder_comb", type=str, default="concat_comb") # sum, concat, concat_comb
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--save_path_summary_gen", type=str, default="save/summary_gen")
parser.add_argument("--model_path", type=str, default="save/test")
parser.add_argument("--save_path_dataset", type=str, default="save/")
parser.add_argument("--cuda", default=True, action="store_true")

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--basic_learner", default=True, action="store_true")
parser.add_argument("--project", action="store_true")
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--softmax", default=True, action="store_true")
parser.add_argument("--mean_query", action="store_true")
parser.add_argument("--schedule", type=float, default=10000)


parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", default=True, action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", default=True, action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="cem")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", default=True, action="store_true")
parser.add_argument("--noam", default=True, action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)

parser.add_argument("--emb_file", type=str)
parser.add_argument("--emb_file_senti", type=str)

## transformer
parser.add_argument("--hop", type=int, default=1)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)
parser.add_argument('--g_pretrain_iterations', type=int, default=1000000, metavar='N',
                    help='steps of pre-training of generators (default: 1000000)')


args = parser.parse_args()
dataset = args.dataset
g_pretrain_iterations = args.g_pretrain_iterations

# Configure training/optimization
clip = 50.0
learning_rate = 1e-6
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

model = args.model
data_dir = args.data_dir
large_decoder = args.large_decoder
topk = args.topk
l1 = args.l1
oracle = args.oracle
basic_learner = args.basic_learner
multitask = args.multitask
softmax = args.softmax
mean_query = args.mean_query
schedule = args.schedule
# Hyperparameters
hidden_dim = args.hidden_dim
emb_dim = args.emb_dim
batch_size = args.batch_size
lr = args.lr
beam_size = args.beam_size
project = args.project
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = args.max_grad_norm
# >>>>>>>>>> OUR ARGS
decoder_comb = args.decoder_comb
'''emo_input = args.emo_input
emo_combine = args.emo_combine
decoder = args.decoder
vae = args.vae
eq6_loss = args.eq6_loss
vader_loss = args.vader_loss'''

init_emo_emb = args.init_emo_emb
pointer_gen = args.pointer_gen

is_coverage = args.is_coverage
#use_oov_emb = args.use_oov_emb
#cov_loss_wt = 1.0
#lr_coverage = 0.15
#eps = 1e-12
#epochs = 10000


USE_CUDA = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

dir_vad = 'dataset/NRC-VAD-Lexicon.txt'

emb_file = args.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))
pretrain_emb = args.pretrain_emb
save_path_summary_gen = args.save_path_summary_gen

save_path = args.save_path
#model_path = args.model_path
save_path_dataset = args.save_path_dataset

test = args.test

### transformer
hop = args.hop
heads = args.heads
depth = args.depth
filter = args.filter


label_smoothing = args.label_smoothing
weight_sharing = args.weight_sharing
noam = args.noam
universal = args.universal
act = args.act
act_loss_weight = args.act_loss_weight


seed = args.seed

if test:
    pretrain_emb = False

collect_stats = False
