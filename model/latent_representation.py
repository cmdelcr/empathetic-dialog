import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

from model.common import init_emo_vad_val, get_vad
from model.emotion_encoder import EmotionInputDecoder
from utils import config
from model.attn_class import zFusion, ComplexResGate, AttnEmo


class VariationalLatentVAD(nn.Module):
    def __init__(self, emotions, input_dim, hidden_dim):
        super(VariationalLatentVAD, self).__init__()
        self.idx2emo = {value: key for key, value in emotions.items()}
        self.vad_emo = init_emo_vad_val(self.idx2emo)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 300
        self.type_input = 'self_att' # self_att, cross_att, att

        self.latent_variables = ['V', 'A', 'D']

        self.decoder_vad = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = EmotionInputDecoder(self.type_input)
            self.decoder_vad[variable] = params_layer

        self.context2params = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = nn.ModuleList([LatentVAD(self.input_dim, self.hidden_dim, self.output_dim) for _ in range(2)])
            self.context2params[variable] = params_layer

        self.params2posterior = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = nn.ModuleList([LatentVAD(self.input_dim*2, self.hidden_dim, self.output_dim) for _ in range(2)])
            self.params2posterior[variable] = params_layer

        self.latent2regression = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = nn.Linear(self.output_dim, 1)
            self.latent2regression[variable] = params_layer


        self.mse_loss = nn.MSELoss(reduction='sum')


    def forward(self, context, mask_src, emo_context_label, enc_batch, mode='prior', vad_rep=None):
        vad_values = get_vad(self.vad_emo, emo_context_label)
        emo_context_label = torch.LongTensor(emo_context_label).to(config.device)
        if mode=='prior':
            context_reduced = context[:, 0]
            latent_params = self.compute_latent_params(context_reduced)
            vad_loss, vad_predicts = self.get_vad_loss(latent_params, vad_values)

            zs = [latent_params[param].z for param in self.latent_variables]
            #z = zs[0] + zs[1] + zs[2]
            #z = self.res_gate(zs[0], zs[1], zs[2])              

            z_context = []
            for variable in self.latent_variables:
                z_context.append(self.decoder_vad[variable](latent_params[variable].z.unsqueeze(1), context, mask_src))
            #z_context = self.decoder_vad(z.unsqueeze(1), context, mask_src)

            return z_context, vad_loss, latent_params
        else:
            x = context[:, 0]
            latent_params = self.compute_latent_params_posterior(
                                x, vad_rep[0][:, 0], vad_rep[1][:, 0], vad_rep[2][:, 0])
            #latent_params = self.compute_latent_params_posterior(
            #                    x, vad_rep.mean(dim=1), None, None)

            return latent_params


    def compute_latent_params(self, context):
        '''Estimate the latent parameters.'''
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])

        for (name, layer) in self.context2params.items():
            mu = layer[0](context)
            logvar = layer[1](context)

            z = self.reparameterize(mu, logvar)
            latent_params[name] = Params(z, mu, logvar)
        
        return latent_params

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        
        return mu + epsilon * std

    def compute_latent_params_posterior(self, context, valence, arousal, dominance):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])

        for (name, layer) in self.params2posterior.items():
            x = torch.cat([context, (valence if name == 'V' else arousal if name == 'A' else dominance)], dim=-1)
            #x = torch.cat([context, valence], dim=-1)
            mu = layer[0](x)
            logvar = layer[1](x)

            z = self.reparameterize(mu, logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params


    def get_vad_loss(self, latent_params, vad_labels):
        predicts = []

        for variable in self.latent_variables:
            logits = self.latent2regression[variable](latent_params[variable].z)
            logits = torch.sigmoid(logits)
            predicts.append(logits)
        predicts = torch.cat(predicts, dim=-1)
        return self.mse_loss(predicts, vad_labels)/config.batch_size, predicts


    def compute_kl_divergence_losses(self, latent_params, latent_params_pos=None):
        idv_kls = dict()
        total_kl = 0.0  # scalar for logging
        total_weighted_kl = torch.tensor(0.0).to(config.device)
        for (latent_name, latent_params) in latent_params.items():
            if latent_params_pos == None:
                kl = self.kl_div(latent_params.mu, latent_params.logvar)
            else:
                kl = self.kl_div(latent_params_pos[latent_name].mu, latent_params_pos[latent_name].logvar, \
                            latent_params.mu, latent_params.logvar)

            total_weighted_kl += kl

        return total_weighted_kl


    def kl_divergence(self, mu, logvar):
        return 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - 1 - logvar)


    def kl_div(self, mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0]).to(config.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(config.device)
            logvar_prior = torch.FloatTensor([0.0]).to(config.device)

        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + \
                 (logvar_posterior.exp() + (mu_posterior - mu_prior).pow(2))/logvar_prior.exp() - one))

        return kl_div


        '''mu_prior = torch.FloatTensor([0.0]).to(config.device)
        logvar_prior = torch.FloatTensor([0.0]).to(config.device)

        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + \
                 (logvar_posterior.exp() + (mu_posterior - mu_prior).pow(2))/logvar_prior.exp() - one))'''

class LatentVAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LatentVAD, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.dropout = nn.Dropout(0.4)  # Regularization

    def forward(self, x):
        #x = self.fc2(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = self.fc2(x)
        return x