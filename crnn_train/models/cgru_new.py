# encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)                                      # [T, b, h * 2]
        # print('recurrent: ', recurrent.size())                              # (18, -1, 512)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        # print('t_rec: ', t_rec.size())                                      # (18, -1, 512)
        output = self.embedding(t_rec)
        # print('output: ', output.size())                                    # (18, -1, 512)
        output = output.view(T, b, -1)

        return output


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_crelu=True):
        super(MyConv2D, self).__init__()
        self.use_crelu = use_crelu

        if self.use_crelu is True:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels / 2, kernel_size=(3, 3), padding=1)
            self.bn = nn.BatchNorm2d(num_features=out_channels / 2)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)

        if self.use_crelu is True:
            x = torch.cat((self.relu(x), self.relu(-x)), 1)             # CReLU
        else:
            x = self.relu(x)                                          # ReLU

        output = self.max_pool(x)
        return output


class CGRU(nn.Module):
    def __init__(self, width, height, n_class, n_c=3):
        super(CGRU, self).__init__()
        assert height % 16 == 0, 'imgH has to be a multiple of 16'
        n_base_conv = 32
        n_hide_size = 256

        self.conv_1 = MyConv2D(in_channels=n_c, out_channels=n_base_conv)
        self.conv_2 = MyConv2D(in_channels=n_base_conv, out_channels=n_base_conv * 2)
        self.conv_3 = MyConv2D(in_channels=n_base_conv * 2, out_channels=n_base_conv * 4, use_crelu=False)

        # self.fc = nn.Linear(in_features=n_base_conv * 16, out_features=n_base_conv)
        # self.bn = nn.BatchNorm1d(num_features=18)
        # self.relu = nn.ReLU()

        # self.gru_1 = nn.GRU(input_size=n_base_conv * 16, hidden_size=n_hide_size, bidirectional=True)
        # self.gru_2 = nn.GRU(input_size=n_base_conv * 16, hidden_size=n_hide_size, bidirectional=True)
        # self.gru_1 = nn.LSTM(input_size=n_base_conv * 32, hidden_size=n_hide_size, bidirectional=True)
        # self.gru_2 = nn.LSTM(input_size=n_base_conv * 16, hidden_size=n_hide_size, bidirectional=True)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(nIn=n_base_conv * 24, nHidden=n_hide_size, nOut=n_hide_size),
            BidirectionalLSTM(nIn=n_hide_size, nHidden=n_hide_size, nOut=n_class)
        )

        # self.fc_end = nn.Linear(in_features=n_base_conv * 16, out_features=n_class)
        # self.dropout = nn.Dropout(p=0.25)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        # print('input: ', input.size())      # (-1, 3, 48, 164)
        x = self.conv_1(input)
        # print('conv_1: ', x.size())         # (-1, 32, 23, 81)
        x = self.conv_2(x)
        # print('conv_2: ', x.size())         # (-1, 64, 10, 39)
        x = self.conv_3(x)
        # print('conv_3: ', x.size())         # (-1, 128, 4, 18)

        b, c, h, w = x.size()
        # print('b %d, c %d, h %d, w %d' % (b, c, h, w))  # b -1, c 128, h 4, w 18
        x = x.reshape([-1, w, c * h])
        # print('reshape: ', x.size())        # (-1, 18, 512)
        # x = self.fc(x)
        # # print('fc: ', x.size())           # (-1, 18, 32)
        # x = self.bn(x)
        # # print('bn: ', x.size())           # (-1, 18, 32)
        # x = self.relu(x)

        x = x.permute(1, 0, 2)                # [w, b, c]
        # print('permute: ', x.size())        # (18, -1, 32)

        # x, _ = self.gru_1(x)
        # print('gru_1: ', x.size())          # (18, -1, 512)
        # x, _ = self.gru_2(x)
        # print('gru_2: ', x.size())          # (18, -1, 512)

        # x = self.dropout(x)
        # x = self.fc_end(x)
        # print('fc_end: ', x.size())         # (18, -1, 84)

        x = self.rnn(x)
        output = x
        return output


def softmax_test():
    data = Variable(torch.FloatTensor([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
                                        [0, 5, 5], [0, 5, 5], [0, 5, 5], [0, 5, 5]],
                                       [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                        [0, 5, 5], [0, 5, 5], [0, 5, 5], [0, 5, 5]]]))
    sofmax = nn.Softmax(dim=2)

    print data
    print data.size()
    data = sofmax(data)
    print data


def gru_test():
    print('gru_test')
    model = CGRU(width=164, height=48, n_class=85)
    data = Variable(torch.randn(1, 3, 48, 164))
    print('input: ', data.size())
    output = model(data)
    print('output: ', output.size())
    # print(output)


if __name__ == '__main__':
    # softmax_test()

    gru_test()
