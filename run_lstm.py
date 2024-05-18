import argparse
import torch
import time
from utils.tools import setup_seed
from exp.exp_main import Exp_vInformer
def initial():
    parser = argparse.ArgumentParser(description="vInformer")
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model',type=str,default='vInformer',help='model name,options:vInformer')
    parser.add_argument('--data',type=str,default='ETTm1',help='dataset type data_dict')
    parser.add_argument('--file_path',type=str,default='./data/ETTm1.csv',help='data file')
    parser.add_argument('--pred_file_path',type=str,default='./data/ETTm1.csv',help='pred_data file')
    parser.add_argument('--checkpoints',type=str,default='./checkpoints/',help='location of model checkpoints')
    parser.add_argument('--root_path', type=str, default='./data_provider', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='tech_cleaned.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ROP', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len',type=int,default=48,help='input sequence length of lstm')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--input_size',type=int,default=2,help='input size of lstm')
    parser.add_argument('--hidden_size',type=int,default=1024,help='hidden size of lstm')
    parser.add_argument('--enc_in', type=int, default=11, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=11, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1,
                        help='output size')  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--split_rate',type=float,default=0.7,help='training-testing split rate')
    parser.add_argument('--n_iters',type=int,default=1000,help='to calculate the number of training epochs')
    parser.add_argument('--train_epochs', type=int, default=31, help='train epochs')#循环次数#epochs 循环次数
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--inverse', action='store_false', default=True,help='inverse output data')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task:WetBulbCelsius')
    parser.add_argument('--freq', type=str, default='15min', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--model_path',type=str,default='./static/modelPath/checkpoint.pth',help='model path')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    start_time = time.time()
    setup_seed(2023)
    args =initial()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('model:',args.model)
    Exp = Exp_vInformer
    for ii in range(args.itr):
        setting = '{}-{}-{}-{}-{}-{}-{}-{}'.format(args.model, args.seq_len,args.pred_len,args.batch_size,args.input_size,args.output_size,args.freq,args.train_epochs)
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    print('总用时：', (time.time() - start_time))
























































