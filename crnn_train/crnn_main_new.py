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
    parser.add_argument('--trainroot', help='path to dataset',default='../crnn/train')
    parser.add_argument('--valroot', help='path to dataset',default='../crnn/val')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    # parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
    parser.add_argument('--img_h', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--img_w', type=int, default=110, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00002, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--crnn', help="path to crnn (to continue training)", default='./save_model/netCRNN.pth')
    # parser.add_argument('--crnn', help="path to crnn (to continue training)", default='')
    parser.add_argument('--alphabet', default=alphabet)
    parser.add_argument('--out_put', help='Where to store samples and models', default='./save_model/netCRNN.pth')
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
                 use_unicode=True, best_acc=0.9):
        self.model = model
        self.use_unicode = use_unicode
        self.converter = utils.strLabelConverter(alphabet)
        self.criterion = CTCLoss()

        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self.transform = T.Compose([
            T.Resize((opt.imgH, opt.imgW)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        train_dataset = dataset.lmdbDataset(root=train_path, transform=self.transform)
        assert train_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchSize,
                                                        shuffle=True, num_workers=int(opt.workers))

        test_dataset = dataset.lmdbDataset(root=test_path, transform=self.transform)
        assert test_dataset
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batchSize,
                                                       shuffle=False, num_workers=int(opt.workers))

    def train_batch(self):
        data = train_iter.next()
        cpu_images, cpu_texts = data  # decode utf-8 to unicode
        if self.use_unicode:
            cpu_texts = [clean_txt(tx.decode('utf-8')) for tx in cpu_texts]

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = self.converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = self.model(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = self.criterion(preds, text, preds_size, length) / batch_size
        self.model.zero_grad()
        cost.backward()
        self.optimizer.step()
        return cost

    def train(self):
        image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
        text = torch.IntTensor(opt.batchSize * 5)
        length = torch.IntTensor(opt.batchSize)

        if opt.cuda:
            crnn.cuda()
            image = image.cuda()
            criterion = self.criterion.cuda()

        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

        # loss averager
        loss_avg = utils.averager()

        # setup optimizer
        if opt.adam:
            optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.adadelta:
            optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
        else:
            optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

        # =================================================================================
        num = 0
        lasttestLoss = 10000
        testLoss = 10000
        numLoss = 0  # 判断训练参数是否下降

        for epoch in range(opt.niter):
            train_iter = iter(train_loader)
            i = 0
            while i < len(train_loader):
                # print('The step{} ........\n'.format(i))
                for p in crnn.parameters():
                    p.requires_grad = True
                crnn.train()
                # if numLoss>50:
                #    cost = trainBatch(crnn, criterion, optimizer,True)
                #    numLoss = 0
                # else:
                cost = self.train_batch(crnn, self.criterion, optimizer)
                loss_avg.add(cost)
                i += 1

                if i % opt.displayInterval == 0:
                    print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
                    loss_avg.reset()

                if i % opt.valInterval == 0:
                    testLoss, accuracy = self.test()
                    # print("[Test] epoch:{}, step:{}, test loss:{}, accuracy:{}".format(epoch, num, testLoss, accuracy))
                    # loss_avg.reset()

                # do checkpointing
                num += 1
                # lasttestLoss = min(lasttestLoss,testLoss)

                if lasttestLoss > testLoss:
                    print("The step {},last lost:{}, current: {},save model!".format(num, lasttestLoss, testLoss))
                    lasttestLoss = testLoss
                    torch.save(crnn.state_dict(), '{}/netCRNN.pth'.format(opt.out_put))
                    numLoss = 0
                else:
                    numLoss += 1

    def test(self):
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        n_correct = 0
        loss_avg = utils.averager()

        time_start = time.time()
        for data, target in self.test_loader:
            cpu_images = data
            cpu_texts = target
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            if self.use_unicode:
                cpu_texts = [clean_txt(tx.decode('utf-8')) for tx in cpu_texts]

            t, l = self.converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = self.model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = self.criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred.strip() == target.strip():
                    n_correct += 1

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))
        accuracy = n_correct / float(len(self.test_loader.dataset))
        test_loss = loss_avg.val()
        loss_avg.reset()
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


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def clean_txt(txt):
    """
    filter char where not in alphabet with ' '
    """
    newTxt = u''
    for t in txt:
        if t in alphabet:
            newTxt += t
        else:
            newTxt += u' '
    return newTxt


def delete(path):
    """
    删除文件
    """
    import os
    import glob
    paths = glob.glob(path+'/*.pth')
    for p in paths:
        os.remove(p)
    

if __name__ == '__main__':
    opt = parse_argvs()

    os.system('mkdir {0}'.format(opt.out_put))

    # opt.manualSeed = random.randint(1, 10000)  # fix seed
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # cudnn.benchmark = True

    # if torch.cuda.is_available() and not opt.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #
    # my_transform = T.Compose([
    #     T.Resize((opt.imgH, opt.imgW)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    # ])
    #
    # train_dataset = dataset.lmdbDataset(root=opt.trainroot, transform=my_transform)
    # assert train_dataset
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=opt.batchSize,
    #     shuffle=True, num_workers=int(opt.workers))
    #
    # test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=my_transform)
    ngpu = int(opt.ngpu)
    nh = int(opt.nh)
    nclass = len(opt.alphabet) + 1
    print("nclass: ", nclass)
    nc = 1

    crnn = crnn.CRNN(opt.imgH, nc, nclass, nh, ngpu)
    crnn.apply(weights_init)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        crnn.load_state_dict(torch.load(opt.crnn))
    print(crnn)

    model_train = ModuleTrain(train_path=opt.trainroot, test_path=opt.valroot, model_file=opt.out_put, model=crnn,
                              img_h=opt.img_h, img_w=opt.img_w, batch_size=opt.batch_size, lr=opt.lr)

    model_train.train()
    model_train.test()

