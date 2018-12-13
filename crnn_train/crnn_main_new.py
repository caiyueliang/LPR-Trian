# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from keys import alphabet
import models.crnn as crnn
import time


def parse_argvs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', help='path to dataset',default='./train')
    parser.add_argument('--valroot', help='path to dataset',default='./val')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    # parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
    parser.add_argument('--img_h', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--img_w', type=int, default=110, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=32, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    # parser.add_argument('--crnn', help="path to crnn (to continue training)", default='./save_model/netCRNN.pth')
    # parser.add_argument('--crnn', help="path to crnn (to continue training)", default='')
    parser.add_argument('--alphabet', default=alphabet)
    parser.add_argument('--out_put', help='Where to store samples and models', default='./checkpoints/crnn.pth')
    parser.add_argument('--use_unicode', type=bool, help='use_unicode', default=True)
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=1000, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    opt = parser.parse_args()
    print(opt)
    return opt


class ModuleTrain:
    def __init__(self, train_path, test_path, model_file, model, img_h=32, img_w=110, batch_size=64, lr=1e-3,
                 use_unicode=True, best_acc=0.9, use_gpu=True, workers=1):
        self.model = model
        self.model_file = model_file
        self.use_unicode = use_unicode
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.lr = lr
        self.best_acc = best_acc
        self.use_gpu = use_gpu
        self.workers = workers

        self.converter = utils.strLabelConverter(alphabet)
        self.criterion = CTCLoss()

        if self.use_gpu:
            print("[use gpu] ...")
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        if torch.cuda.is_available() and not self.use_gpu:
            print("[WARNING] You have a CUDA device, so you should probably run with --cuda")

        # 加载模型
        if os.path.exists(self.model_file):
            self.load(self.model_file)
        else:
            print('[Load model] error !!!')

        self.transform = T.Compose([
            T.Resize((self.img_h, self.img_w)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        train_dataset = dataset.lmdbDataset(root=train_path, transform=self.transform)
        assert train_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=int(self.workers))

        test_dataset = dataset.lmdbDataset(root=test_path, transform=self.transform)
        assert test_dataset
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=int(self.workers))

        # setup optimizer
        # if opt.adam:
        #     self.optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # elif opt.adadelta:
        #     self.optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
        # else:
        #     self.optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epoch, decay_epoch=80):
        image = torch.FloatTensor(self.batch_size, 3, self.img_h, self.img_w)
        text = torch.IntTensor(self.batch_size * 5)
        length = torch.IntTensor(self.batch_size)
        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

        print('[train] epoch: %d' % epoch)
        for epoch_i in range(epoch):
            train_loss = 0.0
            correct = 0

            # if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
            #     self.lr = self.lr * 0.1
            #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            print('================================================')
            for batch_idx, (data, target) in enumerate(self.train_loader):              # 训练
                # data, target = Variable(data), Variable(target)

                if self.use_unicode:
                    target = [tx.decode('utf-8') for tx in target]

                batch_size = data.size(0)
                utils.loadData(image, data)
                t, l = self.converter.encode(target)
                utils.loadData(text, t)
                utils.loadData(length, l)

                if self.use_gpu:
                    image = image.cuda()

                # 梯度清0
                self.optimizer.zero_grad()
                for p in self.model.parameters():
                    p.requires_grad = True
                self.model.train()
                # 计算损失
                preds = self.model(image)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                loss = self.criterion(preds, text, preds_size, length)
                # self.model.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()
                train_loss += loss.item()

                _, preds = preds.max(2)
                # preds = preds.squeeze(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
                for pred, target in zip(sim_preds, target):
                    if pred.strip() == target.strip():
                        correct += 1

            train_loss /= len(self.train_loader.dataset)
            acc = float(correct) / float(len(self.train_loader.dataset))
            print('[Train] Epoch: {} \tLoss: {:.6f}\tAcc: {:.6f}\tlr: {}'.format(epoch_i, train_loss, acc, self.lr))

            # Test
            test_loss, test_acc = self.test()
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                str_list = self.model_file.split('.')
                best_model_file = ""
                for str_index in range(len(str_list)):
                    best_model_file = best_model_file + str_list[str_index]
                    if str_index == (len(str_list) - 2):
                        best_model_file += '_best'
                    if str_index != (len(str_list) - 1):
                        best_model_file += '.'
                self.save(best_model_file)  # 保存最好的模型

        self.save(self.model_file)

    def test(self):
        image = torch.FloatTensor(self.batch_size, 3, self.img_h, self.img_w)
        text = torch.IntTensor(self.batch_size * 5)
        length = torch.IntTensor(self.batch_size)
        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        test_loss = 0.0
        correct = 0
        # loss_avg = utils.averager()

        time_start = time.time()
        for data, target in self.test_loader:
            cpu_images = data
            cpu_texts = target
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            if self.use_unicode:
                cpu_texts = [tx.decode('utf-8') for tx in cpu_texts]

            t, l = self.converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            if self.use_gpu:
                image = image.cuda()

            preds = self.model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            loss = self.criterion(preds, text, preds_size, length)
            test_loss += loss.item()

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred.strip() == target.strip():
                    correct += 1

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))
        accuracy = correct / float(len(self.test_loader.dataset))
        test_loss /= len(self.test_loader.dataset)
        print('[Test] loss: %f, accuray: %f, time: %f' % (test_loss, accuracy, time_avg))
        return test_loss, accuracy

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)
    

if __name__ == '__main__':
    opt = parse_argvs()

    ngpu = int(opt.ngpu)
    nh = int(opt.nh)
    nclass = len(opt.alphabet) + 1
    print("[nclass] ", nclass)
    nc = 1

    crnn = crnn.CRNN(opt.img_h, nc, nclass, nh, ngpu)
    model_train = ModuleTrain(train_path=opt.trainroot, test_path=opt.valroot, model_file=opt.out_put, model=crnn,
                              img_h=opt.img_h, img_w=opt.img_w, batch_size=opt.batch_size, lr=opt.lr)

    model_train.train(200, 80)
    model_train.test()

