# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from train_code.keys import alphabet
import models.crnn as crnn
from train_code import model_train as old_mt
from train_code import model_train_new as new_mt


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

    parser.add_argument('--new_train_mode', type=bool, help='new_train_mode', default=True)
    opt = parser.parse_args()
    print(opt)
    return opt
    

if __name__ == '__main__':
    opt = parse_argvs()

    ngpu = int(opt.ngpu)
    nh = int(opt.nh)
    nclass = len(opt.alphabet) + 1
    print("[nclass] ", nclass)

    if opt.new_train_mode is True:
        nc = 3
        crnn = crnn.CRNN(opt.img_h, nc, nclass, nh, ngpu)
        model_train = new_mt.ModuleTrain(train_path='../../Data/car_recognition/train',
                                         test_path='../../Data/car_recognition/test',
                                         model_file=opt.out_put, model=crnn,
                                         img_h=opt.img_h, img_w=opt.img_w, batch_size=opt.batch_size, lr=opt.lr)
    else:
        nc = 1
        crnn = crnn.CRNN(opt.img_h, nc, nclass, nh, ngpu)
        model_train = old_mt.ModuleTrain(train_path=opt.trainroot, test_path=opt.valroot, model_file=opt.out_put, model=crnn,
                                         img_h=opt.img_h, img_w=opt.img_w, batch_size=opt.batch_size, lr=opt.lr)

    model_train.train(200, 80)
    model_train.test()

