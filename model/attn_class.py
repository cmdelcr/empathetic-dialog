### MOST OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
## MINOR CHANGES
# import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config

from model.common import MultiHeadAttention, LayerNorm, _gen_bias_mask, _gen_timing_signal, _get_attn_subsequent_mask

class ComplexEmoAttentionLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(ComplexEmoAttentionLayer, self).__init__()

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, m, m_tilt, attention_weight, mask = inputs
        m_concat = torch.cat((m, m_tilt), dim=1)
        if mask is None:
            mas_src = None
        else:
            mask_src = torch.cat((mask, mask), dim = 2)

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, m_concat, m_concat,
                                                                mask_src)  # Q, K, V

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Can try remove this positionwise feedforward
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, m_concat, attention_weight, mask


class ComplexResDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        #         super(EmotionInputEncoder, self).__init__()
        #         self.universal = universal
        #         self.num_layers = num_layers
        #         self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        #         if(self.universal):
        #             ## for t
        #             self.position_signal = _gen_timing_signal(num_layers, hidden_size)
        #         params =(hidden_size,
        #                  total_key_depth or hidden_size,
        #                  total_value_depth or hidden_size,
        #                  filter_size,
        #                  num_heads,
        #                  _gen_bias_mask(max_length) if use_mask else None,
        #                  layer_dropout,
        #                  attention_dropout,
        #                  relu_dropout)
        #         self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        #         if(self.universal):
        #             self.enc = EmotionInputAttentionLayer(*params)
        #         else:
        #             self.enc = nn.Sequential(*[EmotionInputAttentionLayer(*params) for l in range(num_layers)])
        #         self.layer_norm = LayerNorm(hidden_size)
        #         self.input_dropout = nn.Dropout(input_dropout)
        #         if(config.act):
        #             self.act_fn = ACT_basic(hidden_size)
        #             self.remainders = None
        #             self.n_updates = None
        super(ComplexResDecoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        if (self.universal):
            self.dec = ComplexEmoAttentionLayer(*params)
        else:
            self.dec = nn.Sequential(*[ComplexEmoAttentionLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, m, m_tilt, mask):
        mask_src = mask
        # Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        if (not config.project): x = self.embedding_proj(x)
        if (self.universal):
            if (config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              m,
                                                                              decoding=True)  # TODO can try set decoding to False and write another mode for act function. Or try without act
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    # TODO Remove positional embedding
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, m, m_tilt, [], mask_src))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                y, _, attn_dist, _ = self.dec((x, m, m_tilt, [], mask_src))

            y = self.layer_norm(y)
        return y


class ComplexResGate(nn.Module):
    def __init__(self, embedding_size):
        super(ComplexResGate, self).__init__()
        self.fc1 = nn.Linear(3*embedding_size, 3*embedding_size)
        self.fc2 = nn.Linear(3*embedding_size, embedding_size)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

        #self.mlp = MLP(embedding_size)

    def forward(self, v, a, d):
        vad_concat = torch.cat((v, a, d), dim=-1)
        x = self.fc1(vad_concat)
        z = self.sigmoid(x)
        y = self.fc2(z + vad_concat) #+ context
        #y = self.fc2(z)
        #y = torch.cat((context, y), dim=1)
        #emo_rep = self.mlp(y)

        #return emo_rep
        return y


class zFusion(nn.Module):
    def __init__(self, emb_dim):
        super(zFusion, self).__init__()
        self.fc1 = nn.Linear(3 * emb_dim, 3 * emb_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3 * emb_dim, emb_dim)
    
    def forward(self, z_v, z_a, z_d):
        z_comb = torch.cat((z_v, z_a, z_d), dim=-1)
        z_combined = self.relu(self.fc1(z_comb))
        z_combined = self.fc2(z_combined + z_comb)
        return z_combined

class GateCombContext(nn.Module):
    def __init__(self, hidden_dim):
        super(GateCombContext, self).__init__()

        self.fc1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        x = self.fc2(out + x)

        return x


class GatedVADCombination(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedVADCombination, self).__init__()
        # Linear layers to compute gates
        self.gate_v = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate_a = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate_d = nn.Linear(hidden_dim * 3, hidden_dim)
    
    def forward(self, v, a, d):
        combined = torch.cat((v, a, d), dim=-1)  # Concatenate vectors
        
        # Compute gates
        gate_v = torch.sigmoid(self.gate_v(combined))
        gate_a = torch.sigmoid(self.gate_a(combined))
        gate_d = torch.sigmoid(self.gate_d(combined))
        
        # Normalize the gates to ensure they sum to 1
        gate_sum = gate_v + gate_a + gate_d
        gate_v = gate_v / gate_sum
        gate_a = gate_a / gate_sum
        gate_d = gate_d / gate_sum
        
        # Weighted combination
        combined_vector = gate_v * v + gate_a * a + gate_d * d
        
        return combined_vector

class AttnEmo(nn.Module):
    def __init__(self, embedding_size):
        super(AttnEmo, self).__init__()
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.layer_norm = LayerNorm(embedding_size)
        self.output_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, encoder_outputs, emotion, mask):
        q = self.query(encoder_outputs)
        k = self.key(emotion)
        v = self.value(emotion)

        logits = torch.bmm(q, k.transpose(1, 2))

        mask = mask.bool()
        logits = logits.masked_fill(mask, -1e18)

        weights = F.softmax(logits, dim=-1)
        weights = self.dropout1(weights)

        contexts = torch.bmm(weights, v)
        attn = self.output_linear(contexts)

        attn = self.dropout2(encoder_outputs + attn)
        context = self.layer_norm(attn)
        context = self.dropout3(encoder_outputs + context)
        #context = torch.cat((context, emotion), dim=1)

        return context
