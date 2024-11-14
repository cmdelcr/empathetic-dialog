import os
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from utils import config
from model.common import load_model, make_infinite, evaluate_generator, merge

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from torch.nn.utils.rnn import pad_sequence

from utils.data_loader import prepare_data_seq, get_vad_values
from model.generator_transformer import GeneratorTransformer
from model.common import count_parameters

from tensorboardX import SummaryWriter



class EmoTrainer():
    def __init__(self):
        super().__init__()        
        self.train_data_loader, self.valid_data_loader, _, self.vocab, self.vocab_emo, self.dict_emo = \
                        prepare_data_seq(batch_size=config.batch_size)
        
        self.generator = GeneratorTransformer(self.vocab, self.vocab_emo, emotions=self.dict_emo)
        for n, p in self.generator.named_parameters():
            if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
                if isinstance(p, nn.Linear):
                    if hasattr(p, 'activation') and p.activation == 'relu':
                        kaiming_uniform_(p, nonlinearity='relu')
                    else:  # Default to Kaiming for ReLU and others
                        xavier_uniform_(p)
                #xavier_uniform_(p)

        print("TRAINABLE PARAMETERS",count_parameters(self.generator))
        self.generator.to(config.device)
        self.criterion = nn.CrossEntropyLoss()

        #path to save the model
        if not os.path.exists(config.save_path):
            os.mkdir(config.save_path)

        #path to save training summary 
        if not os.path.exists(config.save_path_summary_gen):
            os.mkdir(config.save_path_summary_gen)


    def pre_train(self, load_for_test=False):
        self.generator.eval()
        if load_for_test:
            print('Loading generator...')
            self.generator, self.current_loss_gen, self.current_ppl_gen, self.current_acc_gen = load_model(\
                        self.generator)
        else:
            if not os.path.exists(os.path.join(config.save_path, "pre_trained_generator")):
                print('Start pre-training generator with MLE...')
                self.train()
                self.test(type_='pre_train')

            else:
                print('Loading pre-trained generator with MLE...')
                self.generator, self.current_loss_gen, self.current_ppl_gen, self.current_acc_gen = load_model(self.generator)
                print('Pre-trained generator\nTraining loss: {:.5f}, ppl: {:.5f}, acc: {:.5f}'.format(\
                            self.current_loss_gen, self.current_ppl_gen, self.current_acc_gen))

        self.generator.eval()


    def train(self):
        self.generator.train()
        check_iter = 2000
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path_summary_gen)
        data_iter = make_infinite(self.train_data_loader)
        weights_best = deepcopy(self.generator.state_dict())

        for n_iter in tqdm(range(config.g_pretrain_iterations)):
            temp = next(data_iter)
            loss, ppl, bce, acc, dir_loss = self.generator.train_one_batch(temp, n_iter)
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
            writer.add_scalars('vad_loss', {'vad_loss_train': dir_loss['vad_loss']}, n_iter)
            writer.add_scalars('kl_loss', {'kl_loss_train': dir_loss['kl_loss']}, n_iter)
            if config.noam:
                writer.add_scalars('lr', {'learning_rate': self.generator.optimizer._rate}, n_iter)

            if ((n_iter + 1) % check_iter == 0):
                self.generator.eval()
                loss_val, ppl_val, bce_val, acc_val = \
                                    evaluate_generator(self.generator, self.valid_data_loader, ty="valid", max_dec_step=50)
                self.generator.train()
                if (ppl_val <= best_ppl):
                    best_ppl = ppl_val
                    patient = 0
                    writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                    writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                    writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                    writer.add_scalars('accuracy', {'acc_valid': acc_val}, n_iter)

                    self.generator.save_model(loss_val, ppl_val, acc_val, n_iter)
                    weights_best = deepcopy(self.generator.state_dict())
                else:
                    patient += 1
                if patient > 2: 
                    break

        self.generator.load_state_dict({name: weights_best[name] for name in weights_best})
        writer.close()



    def test(self, type_='pre_train'):#'pre_train' adv
        _, _, test_data_loader, _, _, _ = prepare_data_seq(batch_size=config.batch_size)
        self.generator.to(config.device)
        self.generator.eval()

        loss_test, ppl_test, bce_test, acc_test, bleu_score_g, distinct_1, distinc_2, ref_results = \
                        evaluate_generator(self.generator, test_data_loader, ty="test", max_dec_step=50, write_summary=True)

        file_summary = os.path.join(config.save_path, type_ + "_output.txt")
        with open(file_summary, 'w') as the_file:
            the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tDistinct_1\tDistinct_2\n")
            the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                            "test", loss_test, ppl_test, acc_test, bleu_score_g, distinct_1, distinc_2))
            for o in ref_results: the_file.write(o)






