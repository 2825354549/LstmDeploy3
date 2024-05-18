# User: 廖宇
# Data Development:2023/8/30 14:39

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    #它实现了论文中的 Distilling 操作，即对 Encoder 的输出进行降采样，减少序列长度，提高计算效率。
    '''ConvLayer(
        (downConv): Conv1d(512,512, kernel_size=(3,)，stride=(1,)，padding=(1,)，padding_mode=circular)
        (norm): BatchNorm1d(512， eps=le-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ELU(alpha=1.0)
        (maxPool): MaxPool1d(kernel_size=3，stride=2，padding=1，dilation=1，cei_mode=False)
        )
    '''
    def __init__(self,input_size,e_size):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.upConv = nn.Conv1d(in_channels=input_size,
                                  out_channels=e_size,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')#padding_mode='circular' 环形填充
        # self.upConv = nn.Conv1d(in_channels=input_size,
        #                         out_channels=e_size,
        #                         kernel_size=2,
        #                         padding=padding,
        #                         padding_mode='circular')
        # self.norm = nn.BatchNorm1d(e_size)
        self.norm = nn.BatchNorm1d(e_size)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.upConv(x.permute(0, 2, 1))
        #换位置张量 x 的维度从 (batch_size, seq_len, d_model) 转置为 (batch_size, d_model, seq_len)，以便进行一维卷积。
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x)
        x = self.leaky_relu(x)
        x = x.transpose(1,2)
        return x

class ConvLayer2(nn.Module):
        # 它实现了论文中的 Distilling 操作，即对 Encoder 的输出进行降采样，减少序列长度，提高计算效率。
        '''ConvLayer(
            (downConv): Conv1d(512,512, kernel_size=(3,)，stride=(1,)，padding=(1,)，padding_mode=circular)
            (norm): BatchNorm1d(512， eps=le-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ELU(alpha=1.0)
            (maxPool): MaxPool1d(kernel_size=3，stride=2，padding=1，dilation=1，cei_mode=False)
            )
        '''

        def __init__(self, input_size, e_size):
            super(ConvLayer2, self).__init__()
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.upConv = nn.Conv1d(in_channels=input_size,
                                    out_channels=e_size,
                                    kernel_size=3,
                                    padding=padding,
                                    # dtype=torch.float16,
            padding_mode = 'circular',
                                    )  # padding=padding padding_mode='circular'padding_mode='circular' 环形填充
            # self.upConv = nn.Conv1d(in_channels=input_size,
            #                         out_channels=e_size,
            #                         kernel_size=2,
            #                         padding=padding,
            #                         padding_mode='circular')
            self.norm4 = nn.BatchNorm1d(e_size)
            self.norm3 = nn.LayerNorm(e_size)
            # self.norm1 = nn.LayerNorm(self.hidden_size)
            # self.activation = nn.ELU()
            # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.leaky_relu = nn.LeakyReLU(0.5)
            self.hidden_size = e_size
            self.d_ff =  4 * self.hidden_size
            self.dropout = 0.2
            self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.hidden_size, kernel_size=1)
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.dropout = nn.Dropout(self.dropout)
            self.activation = F.gelu

        def forward(self, x):
            x = self.upConv(x.permute(0, 2, 1))
            # print(x.shape)
            # 换位置张量 x 的维度从 (batch_size, seq_len, d_model) 转置为 (batch_size, d_model, seq_len)，以便进行一维卷积。
            # x = self.norm(x)
            # x = self.activation(x)
            # x = self.maxPool(x
            x = self.leaky_relu(x.transpose(-1,1))
            # y = x = x + self.dropout(x)

            # print(x.shape)
            # print(x.shape)
            # print(x.shape)
            # x = self.norm3(x)
            # print(x.shape)

            # x = x + self.dropout(x)
            # print(x.shape)
            # y = x = self.norm4(x)

                # y = x
                # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
                # # print(y.shape)
                #
                # y = self.dropout(self.conv2(y).transpose(-1,1))
                #
                # # # # print(y.shape)
                # # x= x.transpose(-1,1)
                # # y = y.transpose(-1,1)
                # y = self.norm2(x+y)
                # # print(y.shape)
                # # y = y.transpose(-1, 1)
                #
                # # y = y.transpose(-1, 1)

            # print(y.shape)
            return x
class TokenEmbedding(nn.Module):
    def __init__(self, input_size, e_size):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=input_size,
                                   out_channels=e_size,
                                    kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class EncoderLayer2(nn.Module):
    '''https://blog.csdn.net/Cyril_KI/article/details/125095225'''
    def __init__(self,input_size=7,
                 hidden_size=256,
                 num_layers=1,
                 dropout=0.2,
                 device=torch.device('cuda:0'),
                 d_ff = None,
                 bidirectional=False,batch_first=True,activation='relu'):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = device
        super(EncoderLayer2, self).__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.d_ff = d_ff or 4 * self.hidden_size
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=self.batch_first,
                            bidirectional=self.bidirectional)

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    # def initial_hidden_state(self,batch_size):
    #     '''(num_layers * num_directions, batch_size, hidden_size )'''
    #     if self.bidirectional==False:
    #         num_directions = 1
    #     else:
    #         num_directions = 2
    #     # h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
    #     # c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
    #     h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size)
    #     c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size)
    #     # print(num_directions)
    #     hidden_state = (h_0, c_0)
    #     return hidden_state
    def forward(self,x,h=None,c=None):
        if self.bidirectional == False:
            num_directions = 1
        else:
            num_directions = 2
        # if self.device == 'cuda':
        h_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # h_0 = h_0.half()
        # c_0 = c_0.half()
        # else:
        # h_0 = torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)
        lstm_out, (h, c) = self.encoder(x, (h_0,c_0))
        # lstm_out = lstm_out[:, -1, :]

        #残差连接
        # x = x + self.dropout(lstm_out)
        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        return lstm_out, h, c
class EncoderLayer(nn.Module):
    '''https://blog.csdn.net/Cyril_KI/article/details/125095225'''
    def __init__(self,
                 hidden_size=256,
                 num_layers=1,
                 dropout=0.2,
                 device=torch.device('cuda:0'),
                 d_ff = None,
                 bidirectional=False,batch_first=True,activation='relu'):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = device
        super(EncoderLayer, self).__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.d_ff = d_ff or 4 * self.hidden_size
        self.encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=self.batch_first,
                            bidirectional=self.bidirectional)

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.activation = nn.LeakyReLU() if activation == 'relu' else F.gelu

    # def initial_hidden_state(self,batch_size):
    #     '''(num_layers * num_directions, batch_size, hidden_size )'''
    #     if self.bidirectional==False:
    #         num_directions = 1
    #     else:
    #         num_directions = 2
    #     # h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
    #     # c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
    #     h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size)
    #     c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size)
    #     # print(num_directions)
    #     hidden_state = (h_0, c_0)
    #     return hidden_state
    def forward(self,x):
        if self.bidirectional == False:
            num_directions = 1
        else:
            num_directions = 2
        h_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # h_0 = torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)
        lstm_out, (h, c) = self.encoder(x, (h_0,c_0))
        # lstm_out = lstm_out[:, -1, :]

        #残差连接
        # x = x + self.dropout(lstm_out)
        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # return self.norm2(x+y), h, c

        x = x + self.dropout(lstm_out)
        y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(y), h, c


class DecoderLayer(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_layers=1,
                 dropout=0.2,
                 d_ff=None,
                 bidirectional=False,
                 batch_first=True,
                 activation='relu'):
        super(DecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.d_ff = d_ff or 4 * self.hidden_size
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=self.batch_first,
                            bidirectional=self.bidirectional)

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, h=None, c=None):
        #单独调试
        if h is None and c is None:
            if self.bidirectional == False:
                num_directions = 1
            else:
                num_directions = 2
            # h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
            # c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
            h= torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
            c= torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)

        lstm_out, (h, c) = self.decoder(x, (h,c))  # print('lstm_out.shape',lstm_out.shape)
        # 残差连接
        x = x + self.dropout(lstm_out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return  self.norm2(x + y), h, c

        # y = x = self.norm1(lstm_out)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        #y = self.
        # y = lstm_out[:, -1, :].unsqueeze(1)
        # return y, h, c
class DecoderLayer2(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_layers=1,
                 dropout=0.2,
                 d_ff=None,
                 bidirectional=False,
                 batch_first=True,
                 activation='relu'):
        super(DecoderLayer2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.d_ff = d_ff or 4 * self.hidden_size
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=self.batch_first,
                            bidirectional=self.bidirectional)

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, h=None, c=None):
        #单独调试
        if h is None and c is None:
            if self.bidirectional == False:
                num_directions = 1
            else:
                num_directions = 2
            # h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
            # c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
            h= torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
            c= torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)

        lstm_out, (h, c) = self.decoder(x, (h,c))  # print('lstm_out.shape',lstm_out.shape)
        # 残差连接
        # x = x + self.dropout(lstm_out)
        y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        return  y, h, c

        # y = x = self.norm1(lstm_out)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        #y = self.
        # y = lstm_out[:, -1, :].unsqueeze(1)
        # return y, h, c


class Seq2Seq2(nn.Module):
    '''https://blog.csdn.net/Cyril_KI/article/details/112739170'''
    def __init__(self,
                 input_size=7,
                 hidden_size=256,
                 num_layers=1,
                 output_size=7,
                 dropout=0.2,
                 seq_len =32,
                 pred_len=12,
                 device=torch.device('cuda:0')):
        super(Seq2Seq2,self).__init__()

        self.outout_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_first = True
        self.bidirectional = False

        self.upconv = ConvLayer2(self.input_size, self.hidden_size)
        self.encoder = EncoderLayer2(self.hidden_size, self.hidden_size,self.num_layers, self.dropout,device =self.device, d_ff= None
                 ,bidirectional=False,batch_first=True,activation='relu')
        self.decoder = DecoderLayer2(self.hidden_size, self.num_layers, self.dropout,d_ff = None,bidirectional=False,batch_first=True,activation='relu')
        self.norm = nn.LayerNorm(self.hidden_size)
        self.projiection = nn.Sequential(
                                 nn.Linear(self.hidden_size, self.outout_size))
                                 # )nn.GELU(),
        self.down = nn.Sequential(nn.Conv1d(in_channels=self.seq_len,out_channels=self.pred_len,kernel_size=1),nn.ReLU())
        self.relu = nn.ReLU()

        # self.relu = F.relu

    def forward(self,x):

        # if self.bidirectional == False:
        #     num_directions = 1
        # else:
        #     num_directions = 2
        # h_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # h_0 = torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)

        con_y = self.upconv(x)
        # enc_y,h_0,c_0 = self.encoder(con_y)
        # outputs = torch.zeros(x.size(0),self.pred_len,self.hidden_size).to(self.device)
        # # dec_input = enc_y[:, -1, :].unsqueeze(1)
        # # dec_input = outputs[:,-1,:].unsqueeze(1)
        # dec_input  = enc_y
        # for time_step in range(self.pred_len):
        #     dec_y,h,c = self.decoder(dec_input,h_0,c_0)
        #     outputs = torch.cat((outputs,dec_y),dim=1)
        #     dec_input =outputs[:,-1,:].unsqueeze(1)
        #     outputs = torch.cat((dec_y,dec_input),dim =1)
        #     dec_input =  torch.cat((dec_y,dec_input),dim =1)[:,-self.pred_len:,:]
        # # print(outputs.shape)
        # input = self.norm(con_y)

        enc_y, h_0, c_0 = self.encoder(con_y)
        # output_y = torch.cat((input,enc_y),dim=1)
        #     input = output_y[:,-self.pred_len:,:]
        # print(outputs.shape)
        # print(enc_y.shape)
        y = enc_y[:,-self.pred_len:,:]
        # y = self.relu(self.down(x.permute(-1,1))).transpose(-1,1)


        # print(y.shape)
        # y = self.down(enc_y)
        # print(y.shape)
        # print(y.shape)
        # y = self.relu(y).permute(0,2,1)
        y = self.projiection(y)
        # print(y.shape)
        return y
        # return enc_y
class Seq2Seq3(nn.Module):
    '''https://blog.csdn.net/Cyril_KI/article/details/112739170'''
    def __init__(self,
                 input_size=7,
                 hidden_size=256,
                 num_layers=1,
                 output_size=7,
                 dropout=0.2,
                 seq_len =32,
                 pred_len=12,
                 device=torch.device('cuda:0')):
        super(Seq2Seq3,self).__init__()

        self.outout_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_first = True
        self.bidirectional = False

        self.upconv = ConvLayer2(self.input_size, self.hidden_size)
        self.encoder = EncoderLayer2(self.input_size, self.hidden_size,self.num_layers, self.dropout,device =self.device, d_ff= None
                 ,bidirectional=False,batch_first=True,activation='relu')
        self.decoder = DecoderLayer2(self.hidden_size, self.num_layers, self.dropout,d_ff = None,bidirectional=False,batch_first=True,activation='relu')
        self.norm = nn.LayerNorm(self.hidden_size)
        self.projiection = nn.Sequential(
                                 nn.Linear(self.hidden_size, self.outout_size))
                                 # )nn.GELU(),
        self.down = nn.Sequential(nn.Conv1d(in_channels=self.seq_len,out_channels=self.pred_len,kernel_size=1),nn.ReLU())
        self.relu = nn.ReLU()

        # self.relu = F.relu

    def forward(self,x):

        # if self.bidirectional == False:
        #     num_directions = 1
        # else:
        #     num_directions = 2
        # h_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size).to(self.device)
        # h_0 = torch.randn(self.num_layers * num_directions,x.shape[0], self.hidden_size)
        # c_0 = torch.randn(self.num_layers * num_directions, x.shape[0], self.hidden_size)

        # con_y = self.upconv(x)
        # enc_y,h_0,c_0 = self.encoder(con_y)
        # outputs = torch.zeros(x.size(0),self.pred_len,self.hidden_size).to(self.device)
        # # dec_input = enc_y[:, -1, :].unsqueeze(1)
        # # dec_input = outputs[:,-1,:].unsqueeze(1)
        # dec_input  = enc_y
        # for time_step in range(self.pred_len):
        #     dec_y,h,c = self.decoder(dec_input,h_0,c_0)
        #     outputs = torch.cat((outputs,dec_y),dim=1)
        #     dec_input =outputs[:,-1,:].unsqueeze(1)
        #     outputs = torch.cat((dec_y,dec_input),dim =1)
        #     dec_input =  torch.cat((dec_y,dec_input),dim =1)[:,-self.pred_len:,:]
        # # print(outputs.shape)
        # input = self.norm(con_y)

        enc_y, h_0, c_0 = self.encoder(x)
        # output_y = torch.cat((input,enc_y),dim=1)
        #     input = output_y[:,-self.pred_len:,:]
        # print(outputs.shape)
        # print(enc_y.shape)
        y = enc_y[:,-self.pred_len:,:]
        # y = self.relu(self.down(x.permute(-1,1))).transpose(-1,1)


        # print(y.shape)
        # y = self.down(enc_y)
        # print(y.shape)
        # print(y.shape)
        # y = self.relu(y).permute(0,2,1)
        y = self.projiection(y)
        # print(y.shape)
        return y
        # return enc_y

class LSTM(nn.Module):
    def __init__(self,
                 input_size=7,
                 hidden_size=256,
                 num_layers=2,
                 output_size=7,
                 dropout=0.2,
                 seq_len =32,
                 pred_len=12,
                 device=torch.device('cuda:0')
                 ):
        super(LSTM,self).__init__()

        self.outout_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_first = True
        self.bidirectional = False
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout,batch_first=self.batch_first, bidirectional=self.bidirectional)

        self.reg = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.ELU(),
                                 nn.Linear(self.hidden_size, self.outout_size),
                                 )
        self.Linear = nn.Linear(self.seq_len, self.pred_len)


    def initial_hidden_state(self,batch_size):
        '''(num_layers * num_directions, batch_size, hidden_size )'''
        if self.bidirectional==False:
            num_directions = 1
        else:
            num_directions = 2
        h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
        # print(num_directions)
        hidden_state = (h_0, c_0)
        return hidden_state

    def forward(self,x):
        hidden_state = self.initial_hidden_state(x.size(0))
        lstm_out, hidden = self.lstm(x, hidden_state)
        outputs = self.reg(lstm_out)
        # print(outputs.shape)
        outputs = self.Linear(outputs.permute(0,2,1)).permute(0,2,1)

        return outputs

class Seq2Seq(nn.Module):
    '''https://blog.csdn.net/Cyril_KI/article/details/112739170'''
    def __init__(self,
                 input_size=7,
                 hidden_size=256,
                 num_layers=1,
                 output_size=7,
                 dropout=0.2,
                 seq_len =32,
                 pred_len=12,
                 device=torch.device('cuda:0')):
        super(Seq2Seq,self).__init__()
        self.outout_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_first = True
        self.bidirectional = False

        self.upconv = ConvLayer(self.input_size, self.hidden_size)
        self.encoder = EncoderLayer(self.hidden_size, self.num_layers, self.dropout,device =self.device, d_ff= None
                 ,bidirectional=False,batch_first=True,activation='relu')
        self.decoder = DecoderLayer(self.hidden_size, self.num_layers, self.dropout,d_ff = None,bidirectional=False,batch_first=True,activation='relu')

        self.projiection = nn.Sequential(nn.GELU(),
                                 nn.Linear(self.hidden_size, self.outout_size),
                                 )
    def forward(self,x):

        con_y = self.upconv(x)
        enc_y,h_0,c_0 = self.encoder(con_y)
        # outputs = torch.zeros(x.size(0),self.pred_len,self.hidden_size).to(self.device)
        # dec_input = enc_y[:, -1, :].unsqueeze(1)
        # dec_input = outputs[:,-1,:].unsqueeze(1)
        dec_input  = enc_y
        # for time_step in range(self.pred_len):
        #     dec_y,h,c = self.decoder(dec_input,h_0,c_0)
        #     # outputs = torch.cat((outputs,dec_y),dim=1)
        #     # dec_input =outputs[:,-1,:].unsqueeze(1)
        #
        #     # outputs = torch.cat((dec_y,dec_input),dim =1)
        #     dec_input =  torch.cat((dec_y,dec_input),dim =1)[:,-self.pred_len:,:]
        # print(outputs.shape)
        outputs = dec_input[:,-self.pred_len:,:]
        outputs =  self.projiection(outputs)
        return outputs
        # return enc_y


#测试代码 待改
if __name__ == '__main__':
    #测试Seq2Seq
    # model = Seq2Seq2()
    # print(model)
    # x = torch.randn(64,32,7)
    # y = model(x)
    # # print(y[0].shape,y[1].shape)
    # print(y.shape)
    # 测试ConvLayer
    mod = ConvLayer2(256,512)
    input_x =torch.randn(64,32,256)
    out = mod(input_x)
    print(out.shape)

    # 测试encoder
    # model = EncoderLayer()
    # x = torch.randn(64,32,256)
    # y = model(x)
    # print(y[0].shape) #
    # 测试decoder
    # model = DecoderLayer()
    # x = torch.randn(64,1,256)
    # y = model(x)
    # print(y[0].shape)

