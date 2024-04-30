import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

import cv2
from layers.module.reverse_grad import ReverseGrad
from models.resnet import resnet50, embed_net, convDiscrimination, Discrimination
from utils.calc_acc import calc_acc

from layers import TripletLoss, RerankLoss
from layers import CenterTripletLoss
from layers import CenterLoss
from layers import cbam
from layers import NonLocalBlockND
from utils.rerank import re_ranking, pairwise_distance

def intersect1d(tensor1, tensor2):
    return torch.unique(torch.cat([tensor1[tensor1 == val] for val in tensor2]))

def spearman_loss(dist_matrix, rerank_matrix):

    sorted_idx_dist = torch.argsort(dist_matrix, dim=1)
    sorted_idx_rerank = torch.argsort(rerank_matrix, dim=1)

    rank_corr = 0
    n = dist_matrix.size(1)
    for i in range(dist_matrix.size(0)):
        diff = sorted_idx_dist[i] - sorted_idx_rerank[i]
        rank_corr += 1 - (6 * torch.sum(diff * diff) / (n * (n**2 - 1)))

    rank_corr /= dist_matrix.size(0)

    return 1 - rank_corr


def Fb_dt(feat, labels):
    feat_dt = feat
    n_ft = feat_dt.size(0)
    dist_f = torch.pow(feat_dt, 2).sum(dim=1, keepdim=True).expand(n_ft, n_ft)
    dist_f = dist_f + dist_f.t()
    dist_f.addmm_(1, -2, feat_dt, feat_dt.t())
    dist_f = dist_f.clamp(min=1e-12).sqrt()
    mask_ft = labels.expand(n_ft, n_ft).eq(labels.expand(n_ft, n_ft).t())
    mask_ft_1 = torch.ones(n_ft, n_ft, dtype=bool)
    for i in range(n_ft):
        mask_ft_1[i, i] = 0
    mask_ft_2 = []
    for i in range(n_ft):

        mask_ft_2.append(mask_ft[i][mask_ft_1[i]])
    mask_ft_2 = torch.stack(mask_ft_2)
    dist_f_2 = []
    for i in range(n_ft):

        dist_f_2.append(dist_f[i][mask_ft_1[i]])
    dist_f_2 = torch.stack(dist_f_2)
    dist_f_2 = F.softmax(-(dist_f_2 - 1), 1)
    cN_ft = (mask_ft_2[0] == True).sum()
    f_d_ap = []
    for i in range(n_ft):

        f_d_ap.append(dist_f_2[i][mask_ft_2[i]])
    f_d_ap = torch.stack(f_d_ap).flatten()
    loss_f_d_ap = []
    xs_ft = 1
    m_ft = f_d_ap.size(0)
    for i in range(m_ft):
        loss_f_d_ap.append(
            -xs_ft * (1 / cN_ft) * torch.log(xs_ft * cN_ft * f_d_ap[i]))
    loss_f_d_ap = torch.stack(loss_f_d_ap).clamp(max=1e+3).sum() / n_ft
    return loss_f_d_ap


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

def gem_p(x):
    ss = gem(x).squeeze()  # Gem池化
    ss= ss.view(ss.size(0), -1)  # Gem池化
    return ss
def pairwise_dist(x, y):
    # Compute pairwise distance of vectors
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist

def kl_soft_dist(feat1,feat2):
    n_st = feat1.size(0)
    dist_st = pairwise_dist(feat1, feat2)
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)
    for i in range(n_st):  # 将同一类样本中自己与自己的距离舍弃
        mask_st_1[i, i] = 0
    dist_st_2 = []
    for i in range(n_st):
        dist_st_2.append(dist_st[i][mask_st_1[i]])
    dist_st_2 = torch.stack(dist_st_2)
    return dist_st_2


def Bg_kl(logits1, logits2):####输入:(60,206),(60,206)
    KL = nn.KLDivLoss(reduction='batchmean')
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21
    return kl_loss_12, bg_loss_kl
def Sm_kl(logits1, logits2, labels):
    KL = nn.KLDivLoss(reduction='batchmean')
    m_kl = torch.div((labels == labels[0]).sum(), 2, rounding_mode='floor')
    v_logits_s = logits1.split(m_kl, 0)
    i_logits_s = logits2.split(m_kl, 0)
    sm_v_logits = torch.cat(v_logits_s, 1)  # .t()  # 5,206*12->206*12,5
    sm_i_logits = torch.cat(i_logits_s, 1)  # .t()
    sm_kl_loss_vi = KL(F.log_softmax(sm_v_logits, 1), F.softmax(sm_i_logits, 1))
    sm_kl_loss_iv = KL(F.log_softmax(sm_i_logits, 1), F.softmax(sm_v_logits, 1))
    sm_kl_loss = sm_kl_loss_vi + sm_kl_loss_iv
    return sm_kl_loss_vi, sm_kl_loss


def samplewise_entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropies = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropies


def entropy_margin_loss(logits1, logits2, margin):
    entropy1 = samplewise_entropy(logits1)
    entropy2 = samplewise_entropy(logits2)
    losses = torch.exp(F.relu(entropy2 - entropy1 + margin)) - 1
    return losses.mean()


def compute_centroid_distance(features, labels, modalities):
    """
    计算每个类别不同模态的中心特征的距离。

    参数:
    features -- 特征矩阵，形状为(B, C)。
    labels -- 类别标签，形状为(B,)。
    modalities -- 模态标签，形状为(B,)。

    返回:
    distances -- 每个类别模态中心距离的列表。
    """
    unique_labels = torch.unique(labels)
    distances = []
    for label in unique_labels:
        # 分别获取当前类别下的两种模态的特征
        features_modality_0 = features[(labels == label) & (modalities == 0)]
        features_modality_1 = features[(labels == label) & (modalities == 1)]

        # 计算中心特征
        centroid_modality_0 = features_modality_0.mean(dim=0)
        centroid_modality_1 = features_modality_1.mean(dim=0)

        # 计算两个中心特征之间的距离，这里使用欧氏距离
        distance = F.pairwise_distance(centroid_modality_0.unsqueeze(0), centroid_modality_1.unsqueeze(0))
        distances.append(distance)


    return torch.stack(distances)


def modal_centroid_loss(F1, F2, labels, modalities, margin):
    """
    计算损失函数，要求F2中每个类别不同模态的中心距离比F1更小，并施加一个margin。

    参数:
    F1 -- 第一组特征，形状为(B, C)。
    F2 -- 第二组特征，经过网络结构优化，形状为(B, C)。
    labels -- 类别标签，形状为(B,)。
    modalities -- 模态标签，形状为(B,)。
    margin -- 施加的margin值。

    返回:
    loss -- 计算的损失值。
    """
    # 计算F1和F2的中心距离
    distances_F1 = compute_centroid_distance(F1, labels, modalities)
    distances_F2 = compute_centroid_distance(F2, labels, modalities)

    # 计算带margin的损失
    losses = F.relu(distances_F2 - distances_F1 + margin)

    # 返回损失的平均值
    return losses.mean()
class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.decompose = decompose
        self.backbone = embed_net(drop_last_stride=drop_last_stride, decompose=decompose)

        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)


        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck_sp = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck_sp.bias, 0)
        self.bn_neck_sp.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)
        self.CSA1 = kwargs.get('bg_kl', False)
        self.CSA2 = kwargs.get('sm_kl', False)
        self.TGSA = kwargs.get('distalign', False)
        self.IP = kwargs.get('IP', False)
        self.fb_dt = kwargs.get('fb_dt', False)

        if self.decompose:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_sp = nn.Linear(self.base_dim, num_classes, bias=False)
            self.D_special = Discrimination()
            self.C_sp_f = nn.Linear(self.base_dim, num_classes, bias=False)

            self.D_shared_pseu = Discrimination(2048)
            self.grl = ReverseGrad()

        else:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        if self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)
            self.rerank_loss = RerankLoss(margin=0.7)
        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)

    def forward(self, inputs, labels=None, **kwargs):

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        #epoch = kwargs.get('epoch')
        # CNN
        sh_feat, sh_pl, sp_pl, sp_IN,sp_IN_p,x_sp_f,x_sp_f_p = self.backbone(inputs)


        feats = sh_pl

        if not self.training:
            if feats.size(0) == 2048:
                feats = self.bn_neck(feats.permute(1, 0))
                logits = self.classifier(feats)
                return logits  # feats #


            else:
                feats = self.bn_neck(
                    feats)
                return feats

        else:
            return self.train_forward(feats, sp_pl, labels,
                                       sub, sp_IN,sp_IN_p,x_sp_f,x_sp_f_p, **kwargs)



    def train_forward(self, feat, sp_pl, labels,
                       sub, sp_IN,sp_IN_p,x_sp_f,x_sp_f_p, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        loss = 0

        if self.triplet:

            triplet_loss, dist, sh_ap, sh_an = self.triplet_loss(feat.float(), labels)
            triplet_loss_im, _, sp_ap, sp_an = self.triplet_loss(sp_pl.float(), labels)
            trip_loss = triplet_loss + triplet_loss_im
            loss += trip_loss
            metric.update({'tri': trip_loss.data})


        bb = 120  #90
        if self.TGSA:

            sf_sp_dist_v = kl_soft_dist(sp_pl[sub == 0], sp_pl[sub == 0])
            sf_sp_dist_i = kl_soft_dist(sp_pl[sub == 1], sp_pl[sub == 1])
            sf_sh_dist_v = kl_soft_dist(feat[sub == 0], feat[sub == 0])
            sf_sh_dist_i = kl_soft_dist(feat[sub == 1], feat[sub == 1])
            half_B0 = feat[sub == 0].shape[0] // 2
            feat_half0 = feat[sub == 0][:half_B0]
            half_B1 = feat[sub == 1].shape[0] // 2
            feat_half1 = feat[sub == 1][:half_B1]
            feat_cross = torch.cat((feat_half0, feat_half1), dim=0)
            sf_sh_dist_vi = kl_soft_dist(feat_cross, feat_cross)



            _, kl_inter_v = Bg_kl(sf_sh_dist_v, sf_sp_dist_v)
            _, kl_inter_i = Bg_kl(sf_sh_dist_i, sf_sp_dist_i)


            _, kl_intra1 = Bg_kl(sf_sh_dist_v, sf_sh_dist_i)
            _, kl_intra2 = Bg_kl(sf_sh_dist_v, sf_sh_dist_vi)
            _, kl_intra3 = Bg_kl(sf_sh_dist_vi, sf_sh_dist_i)

            kl_intra = kl_intra1 + kl_intra2 + kl_intra3



            if feat.size(0) == bb:
                soft_dt = kl_intra + (kl_inter_v + kl_inter_i) * 0.6


            else:
                soft_dt = (kl_intra1 + kl_inter_v + kl_inter_i) * 0.1

            loss += soft_dt
            metric.update({'soft_dt': soft_dt.data})

        if self.center_loss:
            center_loss = self.center_loss(feat.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})

        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feat.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})


        if self.fb_dt:
            loss_f_d_ap = Fb_dt(feat, labels)
            loss_Fb_im = Fb_dt(sp_pl, labels)
            fb_loss = loss_f_d_ap + loss_Fb_im
            loss += fb_loss

            metric.update({'f_dt': fb_loss.data})

        feat = self.bn_neck(feat)
        sp_pl = self.bn_neck_sp(sp_pl)
        sub_nb = sub + 0  ##模态标签

        if self.IP:
            ################
            ################
            l_F = self.C_sp_f(gem_p(x_sp_f))
            l_F_p = self.C_sp_f(gem_p(x_sp_f_p))
            loss_F = entropy_margin_loss(l_F, l_F_p, 0)
            loss_m_IN = modal_centroid_loss(gem_p(sp_IN), gem_p(sp_IN_p), labels, sub, 0)

            loss += 0.1 * (loss_F + loss_m_IN)
            metric.update({'IN_p': loss_m_IN.data})
            metric.update({'F_p': loss_F.data})

            ################
            ################

        if self.decompose:
            logits_sp = self.classifier_sp(sp_pl)  # self.bn_neck_un(sp_pl)
            loss_id_sp = self.id_loss(logits_sp.float(), labels)


            sp_logits = self.D_special(sp_pl)
            unad_loss_b = self.id_loss(sp_logits.float(), sub_nb)
            unad_loss = unad_loss_b


            pseu_sh_logits = self.D_shared_pseu(feat)
            p_sub = sub_nb.chunk(2)[0].repeat_interleave(2)
            pp_sub = torch.roll(p_sub, -1)
            pseu_loss = self.id_loss(pseu_sh_logits.float(), pp_sub)

            loss += loss_id_sp + unad_loss + pseu_loss

            metric.update({'unad': unad_loss.data})
            metric.update({'id_pl': loss_id_sp.data})

            metric.update({'pse': pseu_loss.data})




        if self.classification:
            logits = self.classifier(feat)
            if self.CSA1:

                _, inter_bg_v = Bg_kl(logits[sub == 0], logits_sp[sub == 0])
                _, inter_bg_i = Bg_kl(logits[sub == 1], logits_sp[sub == 1])

                _, intra_bg = Bg_kl(logits[sub == 0], logits[sub == 1])


                if feat.size(0) == bb:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.8  # intra_bg + (inter_bg_v + inter_bg_i) * 0.7

                else:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.3
                loss += bg_loss
                metric.update({'bg_kl': bg_loss.data})

            if self.CSA2:
                _, inter_Sm_v = Sm_kl(logits[sub == 0], logits_sp[sub == 0], labels)
                _, inter_Sm_i = Sm_kl(logits[sub == 1], logits_sp[sub == 1], labels)
                inter_Sm = inter_Sm_v + inter_Sm_i
                _, intra_Sm = Sm_kl(logits[sub == 0], logits[sub == 1], labels)

                if feat.size(0) == bb:
                    sm_kl_loss = intra_Sm + inter_Sm * 0.8

                else:
                    sm_kl_loss = intra_Sm + inter_Sm * 0.3
                loss += sm_kl_loss
                metric.update({'sm_kl': sm_kl_loss.data})
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        return loss, metric
