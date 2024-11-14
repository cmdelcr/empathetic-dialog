import os
import re
import numpy as np
import math
from collections import Counter
from utils import config

import os
import re
import numpy as np
import math
from collections import Counter
from utils import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.transformer_layers import Encoder, Decoder, GeneratorStep, MulDecoder
from model.common import (LayerNorm, share_embedding, LabelSmoothing, DynamicLabelSmoothing, NoamOpt, get_input_from_batch, get_output_from_batch)   

from model.latent_representation import VariationalLatentVAD
from model.attn_class import ComplexResGate, GateCombContext, zFusion, GatedVADCombination

from sklearn.metrics import accuracy_score



class GeneratorTransformer(nn.Module):
    def __init__(self, vocab, vocab_emo, emotions, model_file_path=None, is_eval=False, load_optim=False):
        super(GeneratorTransformer, self).__init__()
        self.name = 'generator'
        self.vocab = vocab
        self.iter = 0
        self.vocab_emo = vocab_emo
        self.vocab_size = vocab.n_words
        self.word_freq = np.zeros(self.vocab_size)
        self.num_emotions = len(emotions)
        

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.contex_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                                      total_key_depth=config.depth, total_value_depth=config.depth, 
                                      filter_size=config.filter, universal=config.universal)

        
        self.latent_vad = VariationalLatentVAD(emotions, config.emb_dim, config.hidden_dim)

        self.emotion_embedding = nn.Linear(self.num_emotions, config.emb_dim)

        self.decoder = Decoder(config.emb_dim, config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
        self.vad_decoder = ComplexResGate(config.emb_dim)
        self.context_decoder = GateCombContext(config.emb_dim)

        self.decoder_key = nn.Linear(config.hidden_dim, self.num_emotions, bias=False)
        self.generatorstep = GeneratorStep(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generatorstep.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_loss, running_avg_ppl, running_avg_acc, iter_, type_='pre_trained'):
        state = {
            "iter": iter_,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_loss,
            "current_ppl": running_avg_ppl,
            "current_acc": running_avg_acc,
            "model": self.state_dict(),
        }
        self.iter = iter_
        model_save_path = os.path.join(self.model_dir, type_ + "_generator")
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True, reward=0.0):
        enc_batch = batch["input_batch"]
        enc_batch_ext = batch["input_ext_batch"] 
        emo_batch = batch["emotion_context"]
        enc_emo_batch_ext = batch["emotion_context_ext"]
        dec_batch = batch["target_batch"]

        max_oov_length = len(sorted(batch["oovs"], key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.contex_encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        
        vad_hidden, vad_loss, latent_params = self.latent_vad(encoder_outputs, mask_src, batch['emotion_label'], enc_batch)

        if train:
            latent_params_pos = self.latent_vad(encoder_outputs, mask_src, \
                        batch['emotion_label'], enc_batch, 'posterior', vad_hidden)
            kl_loss = self.latent_vad.compute_kl_divergence_losses(latent_params, latent_params_pos)
        else: 
            kl_loss = self.latent_vad.compute_kl_divergence_losses(latent_params)
        
        dir_loss = {'vad_loss': vad_loss.item(), 'kl_loss': kl_loss.item()}
        

        ## Combine Two Contexts
        vad_rep = self.vad_decoder(vad_hidden[0], vad_hidden[1], vad_hidden[2])
        vad_context = self.context_decoder(torch.cat((encoder_outputs, vad_rep), dim=-1))


        emotion_logit = self.decoder_key(vad_rep[:, 0])
        emotion_label = torch.LongTensor(batch['emotion_label']).to(config.device)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, emotion_label)
        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"], pred_emotion)
        
        
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, 1+tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(dec_emb, vad_context, (mask_src, mask_trg), emotion_context=None)

        ## compute output dist
        logit = self.generatorstep(pre_logit, attn_dist, enc_ext_batch if config.pointer_gen else None,
                               extra_zeros, attn_dist_db=None)

        loss_res = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        # loss_emotion: predicting emotion context
        # loss_res: predicting sentence output
        # vad_loss: predictong vad_values of emotion from z
        # kl_loss: Kullback-Leibler divergence (information loss)
        loss = loss_res + loss_emotion + vad_loss + kl_loss + reward

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc, dir_loss
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_emotion.item(), emotion_acc, dir_loss



    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch = batch["input_batch"]
        enc_batch_ext = batch["input_ext_batch"] 
        emo_batch = batch["emotion_context"]
        enc_emo_batch_ext = batch["emotion_context_ext"]

        max_oov_length = len(sorted(batch["oovs"], key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.contex_encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        

        vad_hidden, vad_loss, latent_params = self.latent_vad(encoder_outputs, mask_src, batch['emotion_label'], enc_batch)

        vad_rep = self.vad_decoder(vad_hidden[0], vad_hidden[1], vad_hidden[2])
        vad_context = self.context_decoder(torch.cat((encoder_outputs, vad_rep), dim=-1))


        emotion_logit = self.decoder_key(vad_rep[:, 0])
        #vad_context = vad_hidden
        
        ys = torch.ones(enc_batch.size(0), 1).fill_(config.SOS_idx).long().to(config.device)
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb),
                    self.embedding_proj_in(vad_context), (mask_src, mask_trg), emotion_context=None)
            else:
                out, attn_dist = self.decoder(ys_emb, vad_context, (mask_src, mask_trg), emotion_context=None)

            prob = self.generatorstep(out, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None,
                               extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim=1)

            decoded_words.append(["<EOS>" if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word).to(config.device)], dim=1).to(config.device)
            ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word).to(config.device))), dim=1)

            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)

        return sent

