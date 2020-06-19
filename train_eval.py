import os
import numpy as np
from scipy.spatial.distance import cdist, pdist, euclidean
from scipy.io import savemat, loadmat
from tqdm import tqdm
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import torch
from torch.optim import lr_scheduler
from shutil import copyfile
from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils import get_optimizer,extract_feature
from metrics import mean_ap, cmc, re_ranking
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        # self.mquery_loader = data.mquery_loader

        self.testset = data.testset
        self.queryset = data.queryset
        # self.mqueryset = data.mqueryset

        self.loss = loss
        # print(self.model.module.parameters())
        self.optimizer = get_optimizer(model)

        self.model = model.to('cuda')
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        # print(self.model)
        self.model.train()
        # for batch, (inputs, labels, _) in enumerate(self.train_loader):
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
        # mqf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap
        #
        # ########################   re rank##########################
        # q_g_dist = np.dot(qf, np.transpose(gf))
        # q_q_dist = np.dot(qf, np.transpose(qf))
        # g_g_dist = np.dot(gf, np.transpose(gf))
        # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        #
        # r, m_ap = rank(dist)
        #
        # print('[With Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))
        #
        #
        # #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        index = np.argsort(dist, axis=1)  # from small to large
        index = index[:, :100] + 1

        file = opt.weight
        file = file.split('/model')[0]
        np.savetxt(os.path.join('weights',file, 'rank.txt'), index, delimiter=' ', fmt='%d')

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        ######################### multi query ##########################
        # dist = cdist(qf, gf)

        # r, m_ap = rank(dist)
        # mq = np.zeros_like(qf, dtype=float)
        # for i in range(len(self.queryset.ids)):
        #     mquery_index1 = np.argwhere(np.asarray(self.mqueryset.ids) == self.queryset.ids[i]).ravel()
        #     # if i==0:
        #     #     print(mquery_index1)
        #     mquery_index2 = np.argwhere(np.asarray(self.mqueryset.cameras) == self.queryset.cameras[i]).ravel()
        #     mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        #     # if i == 0:
        #         # print(mquery_index)
        #         # print(mqf[mquery_index, :].shape)
        #     mq[i] = np.mean(mqf[mquery_index, :], axis=0)
        #     # print(mq[i])
        # dist = cdist(mq, gf)
        # r, m_ap = rank(dist)

        # print('[Multi-query Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))
        #
        # q_g_dist = np.dot(mq, np.transpose(gf))
        # q_q_dist = np.dot(mq, np.transpose(mq))
        # g_g_dist = np.dot(gf, np.transpose(gf))
        # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        #
        # r, m_ap = rank(dist)
        #
        # print('[Multi-query With Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))
        ######################### video ##########################
        # qf, q_pids, q_camids = [], [], []
        # for (imgs, pids, camids) in tqdm(self.query_loader):
        #     # print(batch_idx)
        #     # imgs = imgs.cuda()
        #     b, s, c, h, w = imgs.size()
        #     imgs = imgs.view(b * s, c, h, w)
        #
        #     # end = time.time()
        #     # features = model(imgs)
        #     features = extract_feature(self.model, [(imgs, pids, camids)])
        #     # batch_time.update(time.time() - end)
        #     # print(features.size())
        #     # print(b,s)
        #     features = features.view(b, s, -1)
        #     sub_feature1 = features[:, :5, :]
        #     sub_feature2 = features[:, 5:10, :]
        #     sub_feature3 = features[:, 10:15, :]
        #     # sub_feature4 = features[:, 15:, :]
        #     # if pool == 'avg':
        #     # features = torch.mean(features, 1)
        #     sub_feature1 = torch.mean(sub_feature1, 1, keepdim=True)
        #     sub_feature2 = torch.mean(sub_feature2, 1, keepdim=True)
        #     sub_feature3 = torch.mean(sub_feature3, 1, keepdim=True)
        #     # sub_feature4 = torch.mean(sub_feature4, 1)
        #     features = torch.cat([sub_feature1, sub_feature2, sub_feature3], dim=1)
        #     # else:
        #     #     features, _ = torch.max(features, 1)
        #     # features = features.data.cpu()
        #     qf.append(features)
        #     q_pids.extend(pids)
        #     q_camids.extend(camids)
        # qf = torch.cat(qf, 0).numpy()
        # # print(qf)
        # # print(qf.shape)
        # q_pids = np.asarray(q_pids)
        # q_camids = np.asarray(q_camids)
        # # for i in range(len(q_pids)):
        # #     mquery_index1 = np.argwhere(np.asarray(q_pids) == q_pids[i]).ravel()
        # #     # if i==0:
        # #     #     print(mquery_index1)
        # #     mquery_index2 = np.argwhere(np.asarray(q_camids) == q_camids[i]).ravel()
        # #     mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        # #     print(mquery_index)
        # # print(q_pids)
        # # print(q_camids)
        # print("Extracted features for query set, obtained ({}) matrix".format(qf.shape))
        #
        # gf, g_pids, g_camids = [], [], []
        # for (imgs, pids, camids) in tqdm(self.test_loader):
        #     # print(batch_idx)
        #     # imgs = imgs.cuda()
        #     b, s, c, h, w = imgs.size()
        #     imgs = imgs.view(b * s, c, h, w)
        #
        #     # end = time.time()
        #     # features = model(imgs)
        #     features = extract_feature(self.model, [(imgs, pids, camids)])
        #     # batch_time.update(time.time() - end)
        #     features = features.view(b, s, -1)
        #     sub_feature1 = features[:, :5, :]
        #     sub_feature2 = features[:, 5:10, :]
        #     sub_feature3 = features[:, 10:15, :]
        #     # sub_feature4 = features[:, 15:, :]
        #     # if pool == 'avg':
        #     # features = torch.mean(features, 1)
        #     sub_feature1 = torch.mean(sub_feature1, 1, keepdim=True)
        #     sub_feature2 = torch.mean(sub_feature2, 1, keepdim=True)
        #     sub_feature3 = torch.mean(sub_feature3, 1, keepdim=True)
        #     # sub_feature4 = torch.mean(sub_feature4, 1)
        #     features = torch.cat([sub_feature1, sub_feature2, sub_feature3], dim=1)
        #     # else:
        #     #     features, _ = torch.max(features, 1)
        #     # features = features.data.cpu()
        #     gf.append(features)
        #     g_pids.extend(pids)
        #     g_camids.extend(camids)
        # gf = torch.cat(gf, 0).numpy()
        # g_pids = np.asarray(g_pids)
        # g_camids = np.asarray(g_camids)
        #
        # print("Extracted features for gallery set, obtained ({}) matrix".format(gf.shape))
        # # Save to Matlab for check
        # result = {'gallery_f': gf, 'gallery_label': g_pids, 'gallery_cam': g_camids,
        #           'query_f': qf, 'query_label': q_pids, 'query_cam': q_camids}
        # savemat('pytorch_result.mat', result)
        #
        # result = loadmat('pytorch_result.mat')
        # qf = result['query_f']
        # q_camids = result['query_cam'][0]
        # q_pids = result['query_label'][0]
        # gf = result['gallery_f']
        # g_camids = result['gallery_cam'][0]
        # g_pids = result['gallery_label'][0]
        #
        # # qf = np.random.randn(3368,3,256*13)
        # # gf = np.random.randn(19732,3,256*13)
        # # print(qf[:,0,:].shape)
        # dist00 = cdist(qf[:, 0, :], gf[:, 0, :])
        # dist01 = cdist(qf[:, 0, :], gf[:, 1, :])
        # dist10 = cdist(qf[:, 1, :], gf[:, 0, :])
        # dist11 = cdist(qf[:, 1, :], gf[:, 1, :])
        # dist02 = cdist(qf[:, 0, :], gf[:, 2, :])
        # dist20 = cdist(qf[:, 2, :], gf[:, 0, :])
        # dist12 = cdist(qf[:, 1, :], gf[:, 2, :])
        # dist21 = cdist(qf[:, 2, :], gf[:, 1, :])
        # dist22 = cdist(qf[:, 2, :], gf[:, 2, :])
        # dist = dist00+dist01+dist02+dist10+dist11+dist12+dist20+dist21+dist22
        # # print(dist1.shape)
        # r = cmc(dist, q_pids, g_pids, q_camids, g_camids,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, q_pids, g_pids, q_camids, g_camids)
        # print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))
        # q_g_dist = np.dot(qf, np.transpose(gf))
        # q_q_dist = np.dot(qf, np.transpose(qf))
        # g_g_dist = np.dot(gf, np.transpose(gf))
        # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        # r = cmc(dist, q_pids, g_pids, q_camids, g_camids,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, q_pids, g_pids, q_camids, g_camids)
        #
        # print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))


if __name__ == '__main__':

    data = Data()
    model = MGN()
    loss = Loss()
    reid = Main(model, loss, data)
    os.makedirs('weights', exist_ok=True)
    os.makedirs(os.path.join('weights', opt.save_path), exist_ok=True)

    if opt.mode == 'train':
        opt.weight = opt.save_path
        copyfile('./train_eval.py', os.path.join('weights', opt.save_path, 'train_eval.py'))
        copyfile('./network.py', os.path.join('weights', opt.save_path, 'network.py'))
        copyfile('./loss.py', os.path.join('weights', opt.save_path, 'loss.py'))
        copyfile('./utils.py', os.path.join('weights', opt.save_path, 'utils.py'))
        copyfile('./data.py', os.path.join('weights', opt.save_path, 'data.py'))
        for epoch in range(1, opt.epoch+1):
            print('\nepoch', epoch)
            reid.train()
            if epoch % 50 == 0:
                print('\nstart evaluate')
                reid.evaluate()
            torch.save(model.state_dict(), os.path.join('weights', opt.save_path, 'model_{}.pth'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        reid.evaluate()

