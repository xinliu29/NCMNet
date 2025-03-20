import jittor
from utils import jittor_skew_symmetric
import numpy as np
import jittor.nn as nn


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    #print(x1.shape)
    #print(x1.new_ones((batch_size, num_pts)).shape)
    x1 = jittor.concat([x1, x1.new_ones((batch_size, num_pts,1))], dim=-1).reshape(batch_size, num_pts, 3, 1).cpu()
    x2 = jittor.concat([x2, x2.new_ones((batch_size, num_pts,1))], dim=-1).reshape(batch_size, num_pts, 3, 1).cpu()
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1).cpu()
    x2Fx1 = jittor.nn.matmul(x2.transpose(2, 3), jittor.nn.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = jittor.nn.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = jittor.nn.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))
    return ys

class MatchLoss(object):
    def __init__(self, config):
        self.loss_essential = config.loss_essential
        self.loss_classif = config.loss_classif
        self.use_fundamental = config.use_fundamental
        self.obj_geod_th = config.obj_geod_th
        self.geo_loss_margin = config.geo_loss_margin
        self.loss_essential_init_iter = config.loss_essential_init_iter

    def weight_estimation(self, gt_geod_d, is_pos, ones):
        dis = jittor.abs(gt_geod_d - self.obj_geod_th) / self.obj_geod_th

        weight_p = jittor.exp(-dis)
        weight_p = weight_p * is_pos

        weight_n = ones
        weight_n = weight_n * (1 - is_pos)
        weight = weight_p + weight_n

        return weight

    def run(self, global_step, data, logits, ys, e_hat, y_hat):
        # print(data)
        R_in, t_in, xs, pts_virt, y_in = data['Rs'], data['ts'], data['xs'], data['virtPts'], data['ys']

        # Essential/Fundamental matrix loss
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:, :, 2:]
        geod = batch_episym(pts1_virts, pts2_virts, e_hat[-1])
        # essential_loss = jittor.min(geod,self.geo_loss_margin * geod.new_ones(geod.shape))
        essential_loss = jittor.minimum(geod, self.geo_loss_margin * jittor.ones_like(geod))
        essential_loss = jittor.mean(essential_loss)
        # we do not use the l2 loss, just save the value for convenience
        L2_loss = 0
        classif_loss = 0

        # Classification loss
        # The groundtruth epi sqr
        with jittor.no_grad():
            ones = jittor.ones((xs.shape[0], 1))
        for i in range(len(logits)):
            gt_geod_d = ys[i]
            is_pos = (gt_geod_d < self.obj_geod_th).float()
            is_neg = (gt_geod_d >= self.obj_geod_th).float()
            # 检查并修复形状
            if gt_geod_d.numel() == 0:
                continue
            with jittor.no_grad():
                pos = jittor.sum(is_pos, dim=-1, keepdim=True)
                pos_num = nn.relu(pos - 1) + 1
                neg = jittor.sum(is_neg, dim=-1, keepdim=True)
                neg_num = nn.relu(neg - 1) + 1
                pos_w = neg_num / pos_num
                pos_w = jittor.maximum(pos_w, ones)
                weight = self.weight_estimation(gt_geod_d, is_pos, ones)

                 # 获取目标形状
                target_len = min(weight.shape[1], logits[i].shape[1])

                # 截取或填充到相同长度
                if weight.shape[1] > target_len:
                    weight = weight[:, :target_len]
                    is_pos = is_pos[:, :target_len]
                    pos_w = pos_w[:, :target_len]
                elif weight.shape[1] < target_len:
                    weight = jittor.concat([weight, weight[:, -1:].repeat(1, target_len - weight.shape[1])], dim=1)
                    is_pos = jittor.concat([is_pos, is_pos[:, -1:].repeat(1, target_len - is_pos.shape[1])], dim=1)
                    pos_w = jittor.concat([pos_w, pos_w[:, -1:].repeat(1, target_len - pos_w.shape[1])], dim=1)

                # 确保 logits 也是相同长度
                if logits[i].shape[1] > target_len:
                    logits[i] = logits[i][:, :target_len]

            classif_loss += nn.binary_cross_entropy_with_logits(weight * logits[i], is_pos, pos_weight=pos_w)
        gt_geod_d_all = y_in[:, :, 0]
        is_pos_all = (gt_geod_d_all < self.obj_geod_th).float()
        is_neg_all = (gt_geod_d_all >= self.obj_geod_th).float()

         # 获取目标长度
        target_len = min(y_hat.shape[1], is_pos_all.shape[1])

        # 调整 y_hat 和 is_pos_all 的形状
        y_hat_mask = (y_hat < self.obj_geod_th).float()
        if y_hat_mask.shape[1] > target_len:
            y_hat_mask = y_hat_mask[:, :target_len]
        elif y_hat_mask.shape[1] < target_len:
            y_hat_mask = jittor.concat([y_hat_mask, y_hat_mask[:, -1:].repeat(1, target_len - y_hat_mask.shape[1])], dim=1)

        if is_pos_all.shape[1] > target_len:
            is_pos_all = is_pos_all[:, :target_len]
            is_neg_all = is_neg_all[:, :target_len]
        elif is_pos_all.shape[1] < target_len:
            is_pos_all = jittor.concat([is_pos_all, is_pos_all[:, -1:].repeat(1, target_len - is_pos_all.shape[1])], dim=1)
            is_neg_all = jittor.concat([is_neg_all, is_neg_all[:, -1:].repeat(1, target_len - is_neg_all.shape[1])], dim=1)

        precision = jittor.mean(
            jittor.sum((y_hat < self.obj_geod_th).float() * is_pos_all, dim=1) /
            jittor.sum((y_hat < self.obj_geod_th).float() * (is_pos_all + is_neg_all), dim=1)
        )
        recall = jittor.mean(
            jittor.sum((y_hat < self.obj_geod_th).float() * is_pos_all, dim=1) /
            jittor.sum(is_pos_all, dim=1)
        )

        #ratio = jittor.mean(jittor.sum(is_pos_all, dim=1) /
        #                   (jittor.sum(is_pos_all, dim=1) + jittor.sum(is_neg_all, dim=1)))

        loss = 0
        # Check global_step and add essential loss
        if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
            loss += self.loss_essential * essential_loss
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss

        # return [loss, (self.loss_essential * essential_loss).item(), (self.loss_classif * classif_loss).item(), L2_loss,
        #         precision.item(), recall.item()]
        return [loss,0,0,0,0,0]
