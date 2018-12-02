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
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    # parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=110, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--crnn', help="path to crnn (to continue training)", default='./save_model/netCRNN.pth')
    # parser.add_argument('--crnn', help="path to crnn (to continue training)", default='')
    parser.add_argument('--alphabet', default=alphabet)
    parser.add_argument('--experiment', help='Where to store samples and models', default='./save_model')
    parser.add_argument('--use_unicode', type=bool, help='use_unicode', default=True)
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=1000, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    print(opt)
    return opt


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, dataset, criterion, use_unicode):
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    n_correct = 0
    loss_avg = utils.averager()

    time_start = time.time()
    for data, target in test_loader:
        cpu_images = data
        cpu_texts = target
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        if use_unicode:
            cpu_texts = [clean_txt(tx.decode('utf-8')) for tx in cpu_texts]
        # print(cpu_texts)
        t, l = converter.encode(cpu_texts)
        # print(t)
        # print(l)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        # print(preds)
        # print(preds.shape)
        _, preds = preds.max(2)
        # print(preds)
        # print(preds.shape)
        # preds = preds.squeeze(2)
        # print(preds)
        # print(preds.shape)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # print(preds)
        # print(preds.shape)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        # print(sim_preds)
        # print(cpu_texts)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred.strip() == target.strip():
                n_correct += 1

    time_end = time.time()
    time_avg = float(time_end - time_start) / float(len(test_loader.dataset))
    accuracy = n_correct / float(len(test_loader.dataset))
    test_loss = loss_avg.val()
    loss_avg.reset()
    print('[Test] loss: %f, accuray: %f, time: %f' % (testLoss, accuracy, time_avg))
    return test_loss, accuracy


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


def train_batch(net, criterion, optimizer, use_unicode, flage=False):
    data = train_iter.next()
    cpu_images, cpu_texts = data        # decode utf-8 to unicode
    if use_unicode:
        cpu_texts = [clean_txt(tx.decode('utf-8')) for tx in cpu_texts]
        
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    if flage:
        lr = 0.0001
        optimizer = optim.Adadelta(net.parameters(), lr=lr)
    optimizer.step()
    return cost


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

    if opt.experiment is None:
        opt.experiment = 'expr'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    my_transform = T.Compose([
        T.Resize((opt.imgH, opt.imgW)),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_dataset = dataset.lmdbDataset(root=opt.trainroot, transform=my_transform)
    assert train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True, num_workers=int(opt.workers))

    test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=my_transform)

    ngpu = int(opt.ngpu)
    nh = int(opt.nh)
    alphabet = opt.alphabet
    nclass = len(alphabet) + 1
    print("nclass: ", nclass)
    nc = 1

    # =================================================================================
    converter = utils.strLabelConverter(alphabet)
    criterion = CTCLoss()

    crnn = crnn.CRNN(opt.imgH, nc, nclass, nh, ngpu)
    crnn.apply(weights_init)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        crnn.load_state_dict(torch.load(opt.crnn))
    print(crnn)

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

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
    numLoss = 0                 # 判断训练参数是否下降

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
            cost = train_batch(crnn, criterion, optimizer, use_unicode=opt.use_unicode)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % opt.valInterval == 0:
                testLoss, accuracy = val(crnn, test_dataset, criterion, use_unicode=opt.use_unicode)
                # print("[Test] epoch:{}, step:{}, test loss:{}, accuracy:{}".format(epoch, num, testLoss, accuracy))
                # loss_avg.reset()

            # do checkpointing
            num += 1
            # lasttestLoss = min(lasttestLoss,testLoss)

            if lasttestLoss > testLoss:
                print("The step {},last lost:{}, current: {},save model!".format(num,lasttestLoss,testLoss))
                lasttestLoss = testLoss
                # delete(opt. experiment)               # 删除历史模型
                torch.save(crnn.state_dict(), '{}/netCRNN.pth'.format(opt.experiment))
                numLoss = 0
            else:
                numLoss += 1

