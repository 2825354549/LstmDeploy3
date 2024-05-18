import os
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import random
import csv
from sklearn.preprocessing import StandardScaler
from utils.metrics import metric
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0): #改动
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
def loss_plot(y,  name='./pic/test.png',label = None):
    """
    Loss plot
    """
    plt.figure()
    plt.plot(y, label=label, linewidth=2)
    plt.title(label)
    plt.ylabel(label)
    plt.savefig(name, bbox_inches='tight')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def csv_write(data_path,data_name,data):
    with open( data_path+ data_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)
def write_result(folder_path, setting, mae, mse, rse, corr):
    file_name = folder_path + "result.txt"
    if not os.path.exists(file_name):
        open(file_name, 'w').close()
    f = open(file_name, 'a')
    f.write(setting + "  \n")
    f.write(' mse:{}\n mae:{}\n rse:{}\n corr:{}\n'.format(mse, mae, rse, corr))
    f.write('\n')
    f.write('\n')
    f.close()
class GlobalScaler():
    def __init__(self, file_path=None):
        self.mean = np.array([8.90167939e-16,4.60901219e-13])
        self.std = np.array([2.54950976,2.64575131])
        if file_path is not None:
            self.from_csv(file_path)
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        # print('2')
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    def from_csv(self, file_path):
        df_raw = pd.read_csv(file_path)
        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + ['OT']]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        self.fit(df_data.values)