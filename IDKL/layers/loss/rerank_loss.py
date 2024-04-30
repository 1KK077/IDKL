import torch
from torch import nn
from utils.rerank import pairwise_distance

def intersect1d(tensor1, tensor2):
    return torch.unique(torch.cat([tensor1[tensor1 == val] for val in tensor2]))

# def rerank_vc(feat1, feat2, k1=20, k2=6, lambda_value=0.3, eval_type=True):  #q_feat, g_feat  ############代码结果不知正确与否，但没有增加显存了，aini
#     feats = torch.cat([feat1, feat2], 0)
#
#     dist = torch.cdist(feats, feats)
#     original_dist = dist.clone()
#     all_num = original_dist.shape[0]
#     original_dist = (original_dist / original_dist.max(dim=0, keepdim=True).values).transpose(0, 1)
#
#     V = torch.zeros_like(original_dist)
#
#     query_num = feat1.size(0)
#     if eval_type:
#         max_val = dist.max()
#         dist = torch.cat((dist[:, :query_num], max_val.expand_as(dist[:, query_num:])), dim=1)
#     initial_rank = torch.argsort(dist, dim=1)
#
#     for i in range(all_num):
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = (backward_k_neigh_index == i).nonzero(as_tuple=True)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#
#         for j in k_reciprocal_index:
#             candidate = j
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(round(k1 / 2)) + 1]
#             fi_candidate = (candidate_backward_k_neigh_index == candidate).nonzero(as_tuple=True)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
#                     candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = torch.unique(
#                     torch.cat([k_reciprocal_expansion_index, candidate_k_reciprocal_index], dim=0))
#
#         weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)
#
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = torch.zeros_like(V)
#         for i in range(all_num):
#             V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
#         V = V_qe
#
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append((V[:, i] != 0).nonzero(as_tuple=True)[0])
#
#     jaccard_dist = torch.zeros_like(original_dist)
#
#     for i in range(query_num):
#         temp_min = torch.zeros([1, all_num]).cuda()
#         indNonZero = (V[i, :] != 0).nonzero(as_tuple=True)[0]
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j, val in enumerate(indNonZero):
#             temp_min[0, indImages[j]] += torch.minimum(V[i, val], V[indImages[j], val])
#
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     final_dist = final_dist[:query_num, query_num:]
#
#     return final_dist

def rerank_dist(feat1, feat2, k1=20, k2=6, lambda_value=0.3, eval_type=True):  #q_feat, g_feat

    #with torch.no_grad():
    feats = torch.cat([feat1, feat2], 0)  #######
    dist = pairwise_distance(feats, feats)
    original_dist = dist.clone()  # .detach()  # .clone()
    # import pdb
    # pdb.set_trace()
    all_num = original_dist.shape[0]

    #original_dist = original_dist / torch.max(original_dist, dim=0).values

    original_dist = torch.transpose(original_dist, 0,1)  #.transpose(0, 1)
    V = torch.zeros_like(original_dist)  # .half()


    query_num = feat1.size(0)

    #with torch.no_grad():
    if eval_type:
        # dist[:, query_num:] = dist.max()罪魁祸首
        max_val = dist.max()
        dist = torch.cat((dist[:, :query_num], max_val.expand_as(dist[:, query_num:])), dim=1)
    initial_rank = torch.argsort(dist, dim=1)
    # import pdb
    # pdb.set_trace()



    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in k_reciprocal_index:
            candidate = j.item()
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(round(k1 / 2)) + 1]
            # import pdb
            # pdb.set_trace()
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.unique(
                    torch.cat([k_reciprocal_expansion_index, candidate_k_reciprocal_index], 0))

        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = (weight / torch.sum(weight))  # .half()


    original_dist = original_dist[:query_num, ]
    # print('before')
    # objgraph.show_growth(limit=3)

    if k2 != 1:
        V_qe = torch.zeros_like(V)  # .half()
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe

    invIndex = []
    for i in range(all_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])

    jaccard_dist = torch.zeros_like(original_dist)  # .half()

    # print('after')
    # objgraph.show_growth(limit=3)

    # with torch.no_grad():
    #     for i in range(query_num):
    #         temp_min = torch.zeros([1, all_num], device="cuda")
    #         indNonZero = torch.where(V[i, :] != 0)[0]
    #         indImages = [invIndex[ind] for ind in indNonZero]
    #         for j in range(len(indNonZero)):
    #             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]],
    #                                                                               V[indImages[j], indNonZero[j]])
    #         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    for i in range(query_num):
        temp_min = torch.zeros([1, all_num], device="cuda")  # .half()
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += torch.min(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    # print('before')
    # objgraph.show_growth(limit=3)
    # print("Before:", torch.cuda.memory_allocated())
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    # print("After:", torch.cuda.memory_allocated())
    # import pdb
    # pdb.set_trace()  ####jaccard_dist有细微差异和原来的rerank对比
    # del temp_min, jaccard_dist, V, original_dist,forward_k_neigh_index,backward_k_neigh_index,\
    #     k_reciprocal_expansion_index,V_qe,candidate_k_reciprocal_index,candidate_backward_k_neigh_index,\
    #     candidate_forward_k_neigh_index,k_reciprocal_index,fi,fi_candidate
    # torch.cuda.empty_cache()
    final_dist = final_dist[:query_num, query_num:]
    # import pdb
    # pdb.set_trace()
    # del original_dist, dist
    # torch.cuda.empty_cache()
    return final_dist


class RerankLoss(nn.Module):
    def __init__(self, margin=0.03):
        super(RerankLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
    #def forward(self, inputs1, inputs2, targets):

        #n = inputs1.size(0)
        n = inputs.size(0)
        dist = rerank_dist(inputs, inputs)
        #dist = rerank_dist(inputs1, inputs2)

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        # y = dist_an.data.new()
        # y.resize_as_(dist_an.data)
        # y.fill_(1)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        #prec = dist_an.data > dist_ap.data
        #length = torch.sqrt((inputs * inputs).sum(1)).mean()
        return loss, dist,dist_ap, dist_an