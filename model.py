from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        # self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)

        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb



class SoftClusteredSpatialAttention(nn.Module):
    """
    Soft Clustered Spatial Attention.

    输入:
        x: [B, C, N, 1]

    输出:
        out: [B, C, N, 1]

    核心思想:
        1. 使用可学习 cluster_bank 作为簇原型；
        2. 根据当前节点特征动态计算节点到簇的软分配；
        3. 聚合节点特征得到动态 cluster_state；
        4. 节点从 cluster_state 中读取空间信息；

    """

    def __init__(
        self,
        d_model,
        num_nodes,
        seq_length=1,
        dropout=0.1,
        router_dim=64,
        attn_dim=64,
        num_clusters=None,
        cluster_factor=5,

    ):
        super().__init__()
        assert seq_length == 1, "This module is specialized for T=1."

        self.d_model = d_model
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.router_dim = router_dim
        self.attn_dim = attn_dim


        if num_clusters is None:
            num_clusters = max(
                4,
                int(cluster_factor * math.ceil(math.log(max(num_nodes, 2))))
            )

        self.num_clusters = min(num_nodes, num_clusters)

        # =========================
        # 1. 可学习聚类原型
        # =========================
        # 注意：这里不是最终簇中心，而是用于计算软分配的可学习原型。
        self.cluster_bank = nn.Parameter(torch.empty(self.num_clusters, d_model))
        nn.init.xavier_uniform_(self.cluster_bank)

        self.node_q2 = Conv(d_model)
        self.cluster_k = nn.Linear(d_model, d_model, bias=False)
        self.cluster_v = nn.Linear(d_model, d_model, bias=False)


        self.node_q = Conv(d_model)
        self.node_v = Conv(d_model)

        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)


        self.adaptive_embedding = nn.Parameter(
            torch.empty(1, d_model, num_nodes, 1)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        self.t = nn.Conv2d(d_model//2, d_model,1)

    def forward(self, input, tem_emb):
        """
        x: [B, C, N, 1]
        """
        node_q=self.node_q(input)
        node_v = self.node_v(input)


        # [B, C, N, 1] -> [B, N, C]
        node_q = node_q.squeeze(-1).permute(0, 2, 1).contiguous()

        node_v = node_v.squeeze(-1).permute(0, 2, 1).contiguous()

        # =====================================================
        # Step 2: 节点 -> 簇 的软分配
        # =====================================================



        assign_logits = torch.einsum(
            "bnr,mr->bnm",
            node_q,
            self.cluster_bank
        ) / math.sqrt(self.router_dim)

        assign_prob = torch.softmax(assign_logits, dim=-1)  # [B, N, M]

        # =====================================================
        # Step 3: 根据软分配生成动态簇中心
        # =====================================================

        cluster_context = torch.einsum(
            "bnm,bnc->bmc",
            assign_prob,
            node_v
        )  # [B, M, C]

        # 加权平均，避免节点数变多导致尺度变大
        denom = assign_prob.sum(dim=1).clamp_min(1e-6)  # [B, M]
        cluster_state = cluster_context / denom.unsqueeze(-1)




        qn = self.node_q2(input).squeeze(-1).permute(0, 2, 1).contiguous()          # [B, N, attn_dim]
        kc = self.cluster_k(cluster_state)    # [B, M, C]
        vc = self.cluster_v(cluster_state)    # [B, M, C]

        scores = torch.matmul(kc.transpose(1, 2), vc) / math.sqrt(self.attn_dim)
        node2cluster_attn = torch.softmax(scores, dim=-1)  # [B, N, M]

        out = torch.matmul(qn, node2cluster_attn)  # [B, N, C]

        # [B, N, C] -> [B, C, N, 1]
        out = out.permute(0, 2, 1).unsqueeze(-1)

        out = self.proj(out) + out * self.adaptive_embedding*F.tanh(self.t(tem_emb))



        out = self.norm(out)
        out = self.dropout(out)

        return out






class TrendFluctuationDecomposition(nn.Module):

    def __init__(self, num_nodes=170, alpha_min=0.05, alpha_max=0.95):
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.gate_weight = nn.Parameter(torch.empty(num_nodes, 3))
        self.gate_bias = nn.Parameter(torch.zeros(num_nodes))

        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x):
        """
        x: [B, T, N]
        return:
            seasonal: [B, T, N]
            trend: [B, T, N]
        """
        B, T, N = x.shape
        assert N == self.num_nodes

        prev = x[:, 0, :]
        trends = [prev]

        for t in range(1, T):
            cur = x[:, t, :]

            feat = torch.stack(
                [prev, cur, cur - prev],
                dim=-1
            )  # [B, N, 3]

            alpha_logit = (
                                  feat * self.gate_weight.unsqueeze(0)
                          ).sum(dim=-1) + self.gate_bias.unsqueeze(0)

            alpha = torch.sigmoid(alpha_logit)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha

            prev = alpha * prev + (1.0 - alpha) * cur
            trends.append(prev)

        trend = torch.stack(trends, dim=1)
        fluctuation = x - trend
        return fluctuation,trend


class TemporalEmbedding2(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding2, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, :, -1] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, :, -1]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb.permute(0, 1, 3, 2)








class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.scale = self.head_dim ** -0.5

        # QKV 投影（和官方一样）

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影 + Dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, need_weights=True):
        # --------------------------- 维度处理 ---------------------------
        if self.batch_first:
            # 输入形状: [B, L, C] -> 转成 [B, H, L, D]
            B, Lq, _ = query.shape
            B, Lk, _ = key.shape

            # 投影 Q/K/V
            q = query
            k = self.k_proj(key)
            v = self.v_proj(value)

            # 拆多头
            q = q.reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # --------------------------- 注意力计算 ---------------------------
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Lq, Lk]
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # --------------------------- 输出 ---------------------------
        out = torch.matmul(attn_weight, v)  # [B, H, Lq, D]
        out = out.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        out = self.out_proj(out)

        return out


class AdaptiveTemporalQuery(nn.Module):
    def __init__(self,
                 dim=128,
                 num_time=12,
                 num_memory=4,
                 heads=2,
                 depth=2,

                 dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_time = num_time
        self.num_memory = num_memory

        # 时间位置编码，用于保留历史时间顺序
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, num_time, dim)
        )


        self.memory = nn.Parameter(
            torch.randn(1, 1, dim)
        )
        nn.init.trunc_normal_(self.memory, std=0.02)

        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm_m": nn.LayerNorm(dim),
                "norm_x": nn.LayerNorm(dim),
                "cross_attn": MyMultiheadAttention(
    embed_dim=dim,
    num_heads=heads,
    dropout=dropout,
    batch_first=True
),

            }))



    def forward(self, x):
        """
        x: [B, C, N, T]
        return: [B, C, N, 1]
        """

        b, c, n, t = x.shape


        assert t == self.num_time, \
            f"Expected input length {self.num_time}, but got {t}."

        # [B, C, N, T] -> [B, N, T, C] -> [B*N, T, C]
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)

        # 加时间位置编码
        x = x + self.time_pos_embedding[:, :t, :]

        # 扩展 memory token
        # [1, K, C] -> [B*N, K, C]
        memory = self.memory.expand(b * n, -1, -1)

        # memory token 作为 query，历史序列作为 key/value
        for layer in self.layers:
            q = layer["norm_m"](memory)
            kv = layer["norm_x"](x)

            attn_out= layer["cross_attn"](
                query=q,
                key=kv,
                value=kv,
                need_weights=False
            )

            memory = attn_out

        memory = memory.reshape(b, n, c).permute(0, 2, 1).unsqueeze(-1)

        return memory

class DATQ_SCSA(nn.Module):
    def __init__(
            self,
            device,
            input_dim=3,
            channels=64,
            num_nodes=170,
            input_len=12,
            output_len=12,
            dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358 or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes > 200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)

        self.tconv = AdaptiveTemporalQuery(
            dim=channels // 2,
            num_time=input_len,
            num_memory=4,
            heads=4,
            depth=3,
            dropout=dropout
        )
        self.tconv2 = AdaptiveTemporalQuery(
            dim=channels // 2,
            num_time=input_len,
            num_memory=4,
            heads=4,
            depth=3,
            dropout=dropout
        )
        self.start_conv = nn.Conv2d(1, channels // 2, kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(1, channels // 2, kernel_size=(1, 1))
        self.network_channel = channels * 2

        self.SpatialBlock = SoftClusteredSpatialAttention(
            d_model=256,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=0.1,
            router_dim=64,
            attn_dim=64,
            num_clusters=int(math.log(num_nodes))
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )
        self.fc_st2 = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

        self.decomp = TrendFluctuationDecomposition(num_nodes)


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data

        history_data = history_data.permute(0, 3, 2, 1)
        flow = input_data[:, 0, :, :].permute(0, 2, 1)

        fluctuation_init, trend_init = self.decomp(flow)

        fluctuation_init = self.start_conv(fluctuation_init.permute(0, 2, 1).unsqueeze(1))
        trend_init = self.start_conv2(trend_init.permute(0, 2, 1).unsqueeze(1))



        trend = self.tconv(trend_init)
        fluctuation = self.tconv2(fluctuation_init)




        tem_emb = self.Temb(history_data)

        zt = torch.cat(
            (trend + fluctuation, fluctuation_init[..., -1].unsqueeze(-1) + trend_init[..., -1].unsqueeze(-1),tem_emb),
            dim=1)
        zs=self.SpatialBlock(zt, tem_emb)
        zst = zs + self.fc_st(zt) * F.sigmoid(self.fc_st2(zt))

        prediction = self.regression_layer(zst)

        return prediction