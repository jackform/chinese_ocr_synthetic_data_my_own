'''
    Encoder+Decoder+Attention is implemented from:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))

    def forward(self, input):
        # [n, channel_size, 32, 280] -> [n, 512, 1, 71]
        conv = self.cnn(input)
        return conv


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        # input_size代表输入的每个x的维度, hidden_size代码隐层和输出的维度
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        # hidden_size为输入维度, output_size为输出维度
        # because of the bidirectional setting, the output of LSTM is hidden_size * 2
        # bidirectional把每个time step的前向和后向的输出连接起来了
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input : (times, batch, vec) = (71, 8, 512)

        # recurrent: all_output:(71, 8, 256*2), mean 71 times hi
        # _1: hn:( 2*1(ps, bidirectional), 8, 256)
        # _2: cn:the same with _1
        recurrent, (_1, _2) = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, output_size]
        output = output.view(T, b, -1)
        return output


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 行：output_size: classes_num
        # 列：hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input相当于一个下标序列，embedding以此为行号，返回对应的列
        # print('====================> input <======================')
        # print(input)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # print('==================> embedded <==================')
        # print(embedded.size())
        # print(embedded)

        # print('==================>    cat   <====================')
        # print(hidden[0])
        # print(torch.cat((embedded, hidden[0]), 1).size())

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        # print('==================> attn_weights <==============')
        # print(attn_weights.unsqueeze(1).size())
        # print('==================> encoder_outputs <==============')
        # print(encoder_outputs.permute(1, 0, 2).size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))

        # print('================== attn_applied =======================')
        # print(attn_applied.size())

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        # print('==================    output    =========================')
        # print(output.size())
        output = self.attn_combine(output).unsqueeze(0)
        # print(output.size())

        output = F.relu(output)
        # print(output.size())
        output, hidden = self.gru(output, hidden)
        # print('==================    gru    =========================')
        # print(output.size())
        # print(hidden.size())

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Encoder(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(Encoder, self).__init__()
        self.cnn = CNN(channel_size)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"

        # rnn feature
        conv = conv.squeeze(2)  # [b, c, 1, w] -> [b, c, w]
        conv = conv.permute(2, 0, 1)  # [b, c, w] -> [w, b, c]
        output = self.rnn(conv)
        return output


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
