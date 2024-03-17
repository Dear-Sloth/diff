
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')

from torch import optim
import pickle

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
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
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



def visual(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)
    plt.plot([ind_his[-1], ind_his[-1]+1], [history[-1], true[0]], '-', c='#000000', linewidth=1)
    
    plt.plot(ind_out, true, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        plt.plot(ind_out, mean_pred, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        # print(np.shape(ind_out), np.shape(preds))
        plt.plot(ind_out, preds, '-', label='Prediction', c='r', linewidth=1)  # #FFB733    
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')

    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()

def visual_prob(args, history, true, preds=None, mean_pred=None, label_part=None, name='./pic/test.pdf', prob_pd=None):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    if label_part is not None:
        label_out = list(np.arange(len(history)-len(label_part), len(history)))
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)
    plt.plot([ind_his[-1], ind_his[-1]+1], [history[-1], true[0]], '-', c='#000000', linewidth=1)
    
    plt.plot(ind_out, true, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    if mean_pred is not None:
        plt.plot(ind_out, mean_pred, '-', label='Pred-Trend', c='gray', linewidth=1)
    if preds is not None:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>...", np.shape(preds)) # (10, 96)
        mean = np.mean(preds, axis=0).reshape(-1, 1)
        std = np.std(preds, axis=0).reshape(-1, 1)
        # ind_out = ind_out.reshape(-1, 1)
        
        if args.sample_times > 1:
            ub = mean + std
            lb = mean - std
            new_ind_out = np.expand_dims(np.array(ind_out), axis=1)[:,0]
            print(np.shape(new_ind_out), np.shape(ub), np.shape(lb), np.shape(ind_out), np.shape(mean), np.shape(std))
            plt.fill_between(new_ind_out, ub[:,0], lb[:,0], color="#b9cfe7", edgecolor=None)
        # plt.fill_between(ind_out, mean + std, mean - std, facecolor="gray")
        plt.plot(ind_out, mean, '-', label='Prediction', c='r', linewidth=1)  # #FFB733    
    if label_part is not None:
        plt.plot(label_out, label_part, '-', label='Pred-Label', c='pink', linewidth=1)

    plt.legend()
    plt.tight_layout()
    print(name)
    # plt.show()
    plt.savefig(name, bbox_inches='tight') 
    
    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()


def visual2D(true, preds=None, name=0,j=0,inp=0,outp=0):
    """
    Results visualization
    """


    plt.figure(figsize=(48,10))
    plt.title(f'Value_{name}_Channel{j+1}_{inp}_{outp}_per step')
    plt.plot(true, label='GroundTruth', linewidth=0.8)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=0.8)
    plt.legend()
    directory = 'picv'
    if not os.path.exists(directory):  
        os.makedirs(directory)
    file_path = os.path.join(directory, f'Value_{name}_{j+1}_{inp}_{outp}_per step.png')
    plt.savefig(file_path, bbox_inches='tight')
    
    


    
