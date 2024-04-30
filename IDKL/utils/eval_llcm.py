import os
import logging
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.nn import functional as F
from .rerank import re_ranking, pairwise_distance


def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])

    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


# def eval_llcm(query_feats, q_pids, q_camids, gallery_feats, g_pids, g_camids, max_rank=20, rerank=False):
#     """Evaluation with sysu metric
#     Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
#     """
#     ptr = 0
#     query_feat = np.zeros((nquery, 2048))
#     query_feat_att = np.zeros((nquery, 2048))
#     with torch.no_grad():
#         for batch_idx, (input, label) in enumerate(query_loader):
#             batch_num = input.size(0)
#             input = Variable(input.cuda())
#             feat, feat_att = net(input, input, test_mode[1])
#             query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#             query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
#             ptr = ptr + batch_num
#     distmat = -np.matmul(query_feats.cpu().numpy(), np.transpose(gallery_feats.cpu().numpy()))
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     pred_label = g_pids[indices]
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#
#     # compute cmc curve for each query
#     new_all_cmc = []
#     all_cmc = []
#     all_AP = []
#     all_INP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]
#
#         # remove gallery samples that have the same pid and camid with query
#
#         order = indices[q_idx]
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)
#
#         # compute cmc curve
#         # the cmc calculation is different from standard protocol
#         # we follow the protocol of the author's released code
#         new_cmc = pred_label[q_idx][keep]
#         new_index = np.unique(new_cmc, return_index=True)[1]
#
#         new_cmc = [new_cmc[index] for index in sorted(new_index)]
#
#         new_match = (new_cmc == q_pid).astype(np.int32)
#         new_cmc = new_match.cumsum()
#         new_all_cmc.append(new_cmc[:max_rank])
#
#         orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue
#
#         cmc = orig_cmc.cumsum()
#
#         # compute mINP
#         # refernece Deep Learning for Person Re-identification: A Survey and Outlook
#         pos_idx = np.where(orig_cmc == 1)
#         pos_max_idx = np.max(pos_idx)
#         inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
#         all_INP.append(inp)
#
#         cmc[cmc > 1] = 1
#
#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.
#
#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#
#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
#
#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC
#
#     new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
#     new_all_cmc = new_all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     mINP = np.mean(all_INP)
#     return new_all_cmc, mAP, mINP


def eval_llcm(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, rerank=False):
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)
        # dist_mat = -torch.mm(query_feats, gallery_feats.t())

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0]
    r5 = cmc[4]
    r10 = cmc[9]
    r20 = cmc[19]

    r1 = r1 * 100
    r5 = r5 * 100
    r10 = r10 * 100
    r20 = r20 * 100
    mAP = mAP * 100

    perf = 'r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
