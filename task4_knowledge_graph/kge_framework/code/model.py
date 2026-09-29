import logging
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from datetime import datetime

from dataloader import TestDataset

class TransE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):
        super(TransE, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.lmbda = args.lmbda
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]), 
            requires_grad=False
        )

        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_embedding = nn.Embedding(nentity, self.relation_dim)

        self.init_parameters()

    def init_parameters(self):
        if not self.args.use_init:
            nn.init.xavier_uniform_(self.entity_embedding.weight.data)
            nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def load_embedding(self, init_ent_embs, init_rel_embs):

        init_ent_embs = torch.from_numpy(init_ent_embs)
        init_rel_embs = torch.from_numpy(init_rel_embs)

        if self.args.cuda:
            init_ent_embs = init_ent_embs.cuda()
            init_rel_embs = init_rel_embs.cuda()

        self.entity_embedding.weight.data = init_ent_embs
        self.relation_embedding.weight.data = init_rel_embs
        
        print("Load form Embedding success !")

    def loss(self, positive_score, negative_score):
        if self.args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = F.logsigmoid(positive_score)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        loss = (positive_sample_loss + negative_sample_loss)/2
        if self.args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.entity_embedding.norm(p = 3)**3 + 
                self.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
        return loss

    def forward(self, px, nx, py, ny):

        ph = self.entity_embedding(px[:,0])
        pr = self.relation_embedding(px[:,1])
        pt = self.entity_embedding(px[:,2])
        positive_score = self._calc(ph, pr, pt)

        nh = self.entity_embedding(nx[:,0])
        nr = self.relation_embedding(nx[:,1])
        nt = self.entity_embedding(nx[:,2])
        negative_score = self._calc(nh, nr, nt).reshape([-1, self.args.neg_ratio])

        return self.loss(positive_score, negative_score)

    def _calc(self, h, r, t):
        score = h + r - t
        score = self.gamma.item() - torch.norm(score, p=1, dim=1)
        return score.squeeze()

    def predict(self, x):

        h = self.entity_embedding(x[:, 0])
        r = self.relation_embedding(x[:, 1])
        t = self.entity_embedding(x[:, 2])

        score = self._calc(h, r, t)

        return score


# ===========================================================================
# 以下三个类(RotatE / ConvE)为本次作业新增。
#
# 原框架只提供 TransE 一个模型。新增的两个类**严格遵循同一个接口**,以便直接
# 复用框架已有的负采样、filtered MRR / HITS@K 评测、关系类型拆解等设施:
#
#   __init__(model_name, nentity, nrelation, args)
#   loss(positive_score, negative_score)   -> 标量
#   forward(px, nx, py, ny)                -> 标量(内部调用 loss)
#   predict(x)                             -> [N] 个得分,x 形状 [N, 3]
#
# 注意 `predict` 是**逐三元组**打分的(框架的 test_step 依赖这一点),
# 而不是 ConvE 论文里的 1-N 批量打分。数学上等价,只是不如 1-N 高效。
# ===========================================================================
def _kge_loss(model, positive_score, negative_score):
    """与 TransE.loss 同构的损失:正样本 + 负样本的 log-sigmoid,各占一半。

    负样本部分支持自对抗采样(按得分 softmax 加权),与原实现保持一致,
    这样三个模型之间的对比不会被损失函数的差异污染。
    """
    args = model.args
    if args.negative_adversarial_sampling:
        negative_score = (F.softmax(negative_score * args.adversarial_temperature,
                                    dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

    positive_sample_loss = -F.logsigmoid(positive_score).mean()
    negative_sample_loss = -negative_score.mean()
    loss = (positive_sample_loss + negative_sample_loss) / 2

    if args.regularization != 0.0:
        # N3 正则:对实体与关系嵌入施加 L3 范数惩罚
        regularization = args.regularization * (
            model.entity_embedding.weight.norm(p=3) ** 3
            + model.relation_embedding.weight.norm(p=3) ** 3
        )
        loss = loss + regularization
    return loss


class RotatE(nn.Module):
    """RotatE:把关系建模为**复平面上的旋转**。

    打分函数::

        score(h, r, t) = gamma - || h ∘ r - t ||_1

    其中 ``h, r, t`` 都是复数,``∘`` 是逐元素复乘,``r`` 的模长固定为 1
    (只保留相位,即纯旋转)。实现上把复数按实部/虚部**拼接**成两倍长度的实数
    向量,于是实体维度 = ``2 * hidden_dim``,而关系只存 ``hidden_dim`` 个相位角。

    相比 TransE 的表达能力:TransE 只能表示 1-1 关系(平移),对对称关系
    (如"配偶")会强制 h = t;RotatE 的旋转可以表示对称/反对称/逆关系/组合
    四种关系模式,这是它相对 TransE 的主要优势。
    """

    def __init__(self, model_name, nentity, nrelation, args):
        super().__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)

        # 复数用"实部拼接虚部"表示,所以实体维度是 hidden_dim 的两倍
        self.entity_dim = self.hidden_dim * 2
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        # [框架差异] TransE.__init__ 此处写的是 nn.Embedding(nentity, ...),
        # 应为 nrelation —— 见 README 的说明。新模型按正确写法实现。
        self.relation_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.init_parameters()

    def init_parameters(self):
        if self.args.use_init:
            return

        # 实体:**初始化范围必须按 hidden_dim 反比缩放**。
        #
        # [关键] RotatE 的打分是「每维算模长、再对维度求和」,所以距离的期望值
        # 随维度线性增长。若用 xavier 那类与 sqrt(dim) 成反比的界,100 维下
        # 分量约 ±0.42、求和后距离达 30~80,而 gamma 固定为 9 —— 所有得分都是
        # 大负数,logsigmoid 饱和、梯度消失。实测该版本 MRR 仅 0.2886,
        # 不到 TransE(0.5237)的六成。
        #
        # 官方实现取 embedding_range = (gamma + epsilon) / hidden_dim,
        # 使「距离」与「gamma」处在同一量级。epsilon=2 是官方常量。
        self.embedding_range = (self.args.gamma + 2.0) / self.hidden_dim
        nn.init.uniform_(self.entity_embedding.weight.data,
                         -self.embedding_range, self.embedding_range)

        # 关系存的是**相位角**(不是嵌入向量),必须落在 [-π, π]。
        # 用 xavier 之类的初始化没有意义 —— 角度超出这个范围只是绕圈。
        nn.init.uniform_(self.relation_embedding.weight.data,
                         -3.141592653589793, 3.141592653589793)

    def load_embedding(self, init_ent_embs, init_rel_embs):
        self.entity_embedding.weight.data = torch.from_numpy(init_ent_embs)
        self.relation_embedding.weight.data = torch.from_numpy(init_rel_embs)

    def loss(self, positive_score, negative_score):
        return _kge_loss(self, positive_score, negative_score)

    def _calc(self, h, r, t):
        """h, r, t 形状均为 [B, dim]。返回 [B] 的得分。"""
        # 拆成实部/虚部:entity_dim = 2 * hidden_dim
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)

        # 关系只存相位角,取 cos/sin 得到模长为 1 的复数
        re_r = torch.cos(r)
        im_r = torch.sin(r)

        # 复乘 h ∘ r = (re_h*re_r - im_h*im_r) + i*(re_h*im_r + im_h*re_r)
        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r

        # 距离:先把实部虚部叠成 [2, B, D],对**这一对**取 L2 范数(即每个复数
        # 分量的模),再按维度求和 —— 与 RotatE 官方实现一致(.norm() 默认 p=2)。
        #
        # [实现偏差记录] 最初这里写的是 `.norm(p=1, dim=0)`,即每个分量取
        # |re| + |im|。它让距离被系统性放大(约 √2 倍),而 gamma=9 是固定 margin:
        # 距离普遍超过 9 时得分全为负、logsigmoid 饱和、梯度消失。实测该版本
        # 在 countries_S1 上 MRR 只有 0.2632,不到 TransE(0.5237)的一半。
        score = torch.stack([re_score - re_t, im_score - im_t], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return self.gamma.item() - score

    def forward(self, px, nx, py, ny):
        positive_score = self._calc(self.entity_embedding(px[:, 0]),
                                    self.relation_embedding(px[:, 1]),
                                    self.entity_embedding(px[:, 2]))
        negative_score = self._calc(self.entity_embedding(nx[:, 0]),
                                    self.relation_embedding(nx[:, 1]),
                                    self.entity_embedding(nx[:, 2]))
        negative_score = negative_score.reshape([-1, self.args.neg_ratio])
        return self.loss(positive_score, negative_score)

    def predict(self, x):
        return self._calc(self.entity_embedding(x[:, 0]),
                          self.relation_embedding(x[:, 1]),
                          self.entity_embedding(x[:, 2]))


class ConvE(nn.Module):
    """ConvE:把头实体与关系嵌入拼成 2D 矩阵后做**二维卷积**,再与尾实体做内积。

    流程::

        h [B, d] ─┐
                  ├─ 各自 reshape 成 (H, W) 并上下拼接 -> [B, 1, 2H, W]
        r [B, d] ─┘
                  └─ Conv2d -> BatchNorm -> ReLU -> Dropout -> Flatten -> Linear
                     -> 得到 [B, d] 的特征
        得分 = <特征, t> + bias[t]

    相比 TransE/RotatE 这类"距离型"打分,ConvE 是**相似度型**:卷积能捕捉头实体
    与关系之间的二维局部交互,表达能力更强,代价是参数量大得多(在 countries_S1
    这种只有 2 个关系的小数据集上,这个代价换不来收益 —— 见 README 的分析)。
    """

    def __init__(self, model_name, nentity, nrelation, args):
        super().__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args
        self.hidden_dim = args.hidden_dim

        # 论文中 emb_dim 通常取 200 并 reshape 成 10x20。这里允许任意维度:
        # 优先用 10 作列数(hidden_dim 能被 10 整除时),否则退化为方阵。
        self.emb_w = 10 if self.hidden_dim % 10 == 0 else int(self.hidden_dim ** 0.5)
        self.emb_h = self.hidden_dim // self.emb_w
        if self.emb_h * self.emb_w != self.hidden_dim:
            raise ValueError(
                f"ConvE 需要嵌入维度可reshape为二维,当前 hidden_dim={self.hidden_dim} "
                f"无法分解为整数行列(建议取 10 的倍数,如 100 / 200)")

        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_embedding = nn.Embedding(nrelation, self.relation_dim)
        # 输出端的实体偏置。ConvE 论文用它补偿"尾实体作为分类目标"时的频率偏差,
        # 加不加差别不大,但这是标准配置
        self.entity_bias = nn.Embedding(nentity, 1)

        # 卷积核数量与论文一致取 32;3x3 是论文的默认值
        self.n_filters = 32
        self.conv = nn.Conv2d(1, self.n_filters, (3, 3), 1, 0)
        self.bn = nn.BatchNorm2d(self.n_filters)
        self.dropout = nn.Dropout(0.2)
        # 卷积后的展平长度:H、W 各被 3x3 无 padding 卷积吃掉 2
        flat = self.n_filters * (2 * self.emb_h - 2) * (self.emb_w - 2)
        self.fc = nn.Linear(flat, self.entity_dim)

        self.init_parameters()

    def init_parameters(self):
        if self.args.use_init:
            return
        bound = 6.0 / (self.entity_dim ** 0.5)
        nn.init.uniform_(self.entity_embedding.weight.data, -bound, bound)
        nn.init.uniform_(self.relation_embedding.weight.data, -bound, bound)
        nn.init.zeros_(self.entity_bias.weight.data)

    def load_embedding(self, init_ent_embs, init_rel_embs):
        self.entity_embedding.weight.data = torch.from_numpy(init_ent_embs)
        self.relation_embedding.weight.data = torch.from_numpy(init_rel_embs)

    def loss(self, positive_score, negative_score):
        return _kge_loss(self, positive_score, negative_score)

    def _features(self, h, r):
        """把 (头, 关系) 编码成 [B, d] 的特征向量。"""
        b = h.size(0)
        x = torch.cat([h, r], dim=-1).view(b, 1, 2 * self.emb_h, self.emb_w)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc(x.view(b, -1))

    def _calc(self, h, r, t, t_id=None):
        feat = self._features(h, r)
        score = (feat * t).sum(dim=-1)
        if t_id is not None:
            score = score + self.entity_bias(t_id).squeeze(-1)
        return score

    def forward(self, px, nx, py, ny):
        positive_score = self._calc(self.entity_embedding(px[:, 0]),
                                    self.relation_embedding(px[:, 1]),
                                    self.entity_embedding(px[:, 2]), px[:, 2])
        negative_score = self._calc(self.entity_embedding(nx[:, 0]),
                                    self.relation_embedding(nx[:, 1]),
                                    self.entity_embedding(nx[:, 2]), nx[:, 2])
        negative_score = negative_score.reshape([-1, self.args.neg_ratio])
        return self.loss(positive_score, negative_score)

    def predict(self, x):
        return self._calc(self.entity_embedding(x[:, 0]),
                          self.relation_embedding(x[:, 1]),
                          self.entity_embedding(x[:, 2]), x[:, 2])


def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()
    optimizer.zero_grad()

    positive_batch, negative_batch, yp_batch, yn_batch = next(train_iterator)

    if args.cuda:
        positive_batch = positive_batch.cuda()
        negative_batch = negative_batch.cuda()
        yp_batch = yp_batch.cuda()
        yn_batch = yn_batch.cuda()

    loss = model(positive_batch, negative_batch, yp_batch, yn_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()

    log = {
        'loss': loss.item()
    }

    return log

def test_step(model, valid_triples, test_dataset_list, entity2id, id2entity, relation2id, id2relation, relation2type, args):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    #Prepare dataloader for evaluation

    logs = []
    detail_logs = [[[],[],[],[]], [[],[],[],[]]]
    dd = ["N-N", "N-1", "1-N", "1-1"]
    mm = {'head':0, 'tail':1}

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                if args.cuda:
                    s = "cuda:0"
                    positive_sample = positive_sample.to(s, non_blocking=True)
                    negative_sample = negative_sample.to(s, non_blocking=True)
                    filter_bias = filter_bias.to(s, non_blocking=True)

                batch_size = positive_sample.size(0)

                nentity = negative_sample.size(1)
                score = model.predict(negative_sample.reshape([-1, 3]))
                score = score.reshape([-1, nentity])
                score += filter_bias
                
                argsort = torch.argsort(score, dim = 1, descending=True)
                
                if mode == 'head':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                check_ids = []
                for i in range(batch_size):
                    relation = positive_sample[i][1]
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                    ttype = relation2type[relation.item()]
                    detail_logs[mm[mode]][ttype].append({
                        'MRR_' + dd[ttype] + '_' + mode: 1.0/ranking,
                        'MR_' + dd[ttype] + '_' + mode: float(ranking),
                        'HITS@1_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 1 else 0.0,
                        'HITS@3_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 3 else 0.0,
                        'HITS@10_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 10 else 0.0,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    if not logs:
        return metrics
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    # [健壮性修补] 原代码直接访问 detail_logs[i][j][0],假定 4 种关系类型
    # (N-N / N-1 / 1-N / 1-1)在该数据集中都出现过。但小数据集往往只含少数
    # 关系 —— 例如 countries_S1 只有 2 个关系,大部分桶为空,于是抛
    #   IndexError: list index out of range
    # 导致评测阶段直接崩溃。空桶跳过即可,不影响其余指标的计算。
    for i in range(2):
        for j in range(4):
            if not detail_logs[i][j]:
                continue
            for metric in detail_logs[i][j][0].keys():
                metrics[metric] = sum([log[metric] for log in detail_logs[i][j]])/len(detail_logs[i][j])
    return metrics