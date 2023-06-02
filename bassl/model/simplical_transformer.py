import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import MlpHead

class MultiHeadAttentionSimplicial(nn.Module):
    def __init__(self, d_model, num_heads=8, bias=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.B_weights = torch.nn.Parameter(torch.Tensor(self.head_dim, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.B_weights)

        self.Q = nn.Linear(d_model, d_model)
        self.K1 = nn.Linear(d_model, d_model)
        self.K2 = nn.Linear(d_model, d_model)
        self.V1 = nn.Linear(d_model, d_model)
        self.V2 = nn.Linear(d_model, d_model)

        # self.Q = nn.Linear(2048, d_model)
        # self.K1 = nn.Linear(2048, d_model)
        # self.V1 = nn.Linear(2048, d_model)
        # self.K2 = nn.Linear(90, d_model)
        # self.V2 = nn.Linear(90, d_model)

    def multi_softmax(self, target, axis, name=None):
        max_axis = torch.amax(target, axis, keepdim=True)
        target_exp = torch.exp(target - max_axis)
        normalize = torch.sum(target_exp, axis, keepdim=True)
        softmax = target_exp / normalize
        return softmax

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, y, z):
        B, seq_len, _ = x.shape

        q = self.Q(x)
        k1 = self.K1(y)
        v1 = self.V1(y)
        k2 = self.K2(z)
        v2 = self.V2(z)

        q = self.transpose_for_scores(q)
        k1 = self.transpose_for_scores(k1)
        k2 = self.transpose_for_scores(k2)
        v1 = self.transpose_for_scores(v1)
        v2 = self.transpose_for_scores(v2)

        q = torch.reshape(q, (-1, seq_len, self.head_dim))
        k1 = torch.reshape(k1, (-1, seq_len, self.head_dim))
        k2 = torch.reshape(k2, (-1, seq_len, self.head_dim))
        v1 = torch.reshape(v1, (-1, seq_len, self.head_dim))
        v2 = torch.reshape(v2, (-1, seq_len, self.head_dim))

        qk1 = torch.einsum('aib,ajb->aij', q, k1)
        qk2 = torch.einsum('aib,akb->aik', q, k2)
        k1k2 = torch.einsum('ajb,akb->ajk', k1, k2)

        k2k2 = torch.einsum('akc,akc->ak', k2, k2)
        qq = torch.einsum('aib,aib->ai', q, q)
        k1k1 = torch.einsum('ajb,ajb->aj', k1, k1)

        qk1k2k2 = torch.einsum('aij,ak->aijk', torch.square(qk1), k2k2)
        k1k2qq = torch.einsum('ajk,ai->aijk', torch.square(k1k2), qq)
        qk2k1k1 = torch.einsum('aik,aj->aijk', torch.square(qk2), k1k1)

        qk1_e = torch.unsqueeze(qk1, axis=3)  # qk1_e = tf.einsum('aij->aijk',qk1)
        qk2_e = torch.unsqueeze(qk2, axis=2)  # qk2_e = tf.einsum('aik->aijk',qk2)
        k1k2_e = torch.unsqueeze(k1k2, axis=1)  # k1k2_e = tf.einsum('ajk->aijk',k1k2)

        pre_logitsvector = qk1k2k2 + k1k2qq + qk2k1k1 - 2 * qk1_e * qk2_e * k1k2_e

        # print(torch.min(pre_logitsvector).item(), torch.max(pre_logitsvector).item())

        pre_logitsvector[torch.isinf(pre_logitsvector)] = 55500

        # if torch.isinf(pre_logitsvector.view(-1)).sum().item() > 0:
        #     print(torch.min(pre_logitsvector).item(), torch.max(pre_logitsvector).item())
        #     pre_logitsvector[torch.isinf(pre_logitsvector)] = 55500

        logitsvector = torch.sqrt(pre_logitsvector)

        a = self.multi_softmax(logitsvector, axis=[-2, -1])

        Bvj = torch.einsum('qrs,ajr->aqsj', self.B_weights, v1)
        Bvjvk = torch.einsum('aqsj,aks->aqjk', Bvj, v2)

        attention_heads = torch.einsum('aijk,aqjk->aiq', a, Bvjvk)

        attention_heads = torch.reshape(attention_heads, (-1, self.num_heads, seq_len, self.head_dim))
        #attention_heads = torch.permute(attention_heads, [0, 2, 1, 3])  # (-1, seq_len, self.num_heads, d_simp_model//heads)
        attention_heads = attention_heads.permute(0, 2, 1, 3)
        attention_heads = torch.reshape(attention_heads, (-1, seq_len, self.d_model))

        # of shape (-1, seq_len, d_simp_model)
        return attention_heads

class SimplicalTransformer(nn.Module):
    def __init__(self, d_model, num_heads=8, bias=False):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = MultiHeadAttentionSimplicial(d_model)

        self.emb_x = nn.Linear(2048, d_model)
        self.emb_y = MlpHead(input_dim=2048, hidden_dim=2048, output_dim=d_model)
        self.emb_z = MlpHead(input_dim=90, hidden_dim=512, output_dim=d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, y, z):

        y = y.to(torch.float16)
        z = z.to(torch.float16)

        x = self.emb_x(x)
        y = self.emb_y(y)
        z = self.emb_z(z)

        # print(1, torch.min(x).item(), torch.max(x).item())

        pre = x

        # x = self.norm1(x)
        # y = self.norm2(y)
        # z = self.norm3(z)

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        z = F.normalize(z, dim=-1)

        x = self.cross_attn(x, y, z)
        # x = torch.cat((pre, x), dim=-1)

        x = pre + x

        # if torch.isnan(torch.max(x)):
        #     # print(1, torch.max(pre).item(), torch.max(y).item(), torch.max(z).item())
        #     # print(2, torch.min(pre).item(), torch.min(y).item(), torch.min(z).item())
        #     # print(3, torch.max(x).item(), torch.min(x).item())
        #     print('Nan detected')
        #     x[x != x] = 0
        #     #torch.nan_to_num(x, nan=0., posinf=1.0, neginf=-1.0)

        # x = self.norm4(x)

        return x