import os
import torch
import json
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(
        self,
        model,
        epoch,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        res_root='./res',
        checkpoint_path='./res/checkpoint/',
        loss_path='./res/loss/',
        check_frequence=0,
        training_step=1,
        valid_step=1,
        
    ):
        self.model = model
        self.model.to(device)
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_step = training_step
        self.valid_step = valid_step
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.check_frequence = check_frequence
        self.checkpoint_path = checkpoint_path
        self.log = lambda x: 10*math.log10(x)
        self.loss = {
            'train_mse':[],
            'val_mse':[],
        }
        self.res_root = res_root
        self.loss_path = loss_path
    
    def train(self):
        for e in range(self.epoch):
            self.train_epoch()
            self.valid_epoch()
            print('Epoch:{:n}, Average training loss: {:3f}, Average valid loss: {:3f} \n'.format(
                int(e + 1),
                self.loss['train_mse'][-1], 
                self.loss['val_mse'][-1]
                )
            )
            
            self.lr_scheduler.step()
            if self.check_frequence and not (e + 1)%self.check_frequence:
                self.getCheckpoint()
                
        if not os.path.exists(self.res_root):
            os.mkdir(self.res_root)
        # self.getCheckpoint()
        # self.saveLoss()
        
    
    def train_epoch(self):
        self.model.train()
        running_loss = []
        
        for ind, data in enumerate(self.train_dataloader):
            src, tag = data[0].to(device), data[1].to(device)
            
            self.optimizer.zero_grad()
            out = self.model(src)
            loss = self.criterion(out, tag, reduction='sum')
            loss.backward()
            
            train_batch_mse = self.log(loss.item()/src.shape[0])
            running_loss.append(train_batch_mse)
            self.optimizer.step()

            if ind == self.training_step:
                break
        
        epoch_loss = np.mean(running_loss)
        
        self.loss['train_mse'].append(epoch_loss)

    def valid_epoch(self):
        self.model.eval()
        running_loss = []
        
        with torch.no_grad():
            for ind, data in enumerate(self.val_dataloader):
                src, tag = data[0].to(device), data[1].to(device)
                out = self.model(src)
                loss = self.criterion(out, tag, reduction='sum')
                running_loss.append(self.log(loss.item()/src.shape[0]))

                if ind == self.valid_step:
                    break
        
        epoch_loss = np.mean(running_loss)     
        self.loss['val_mse'].append(epoch_loss)
    
    def getCheckpoint(self,):
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        save_path = os.path.join(self.checkpoint_path, 'model.pt')
        torch.save(self.model, save_path)
    
    def saveLoss(self):
        if not os.path.exists(self.loss_path):
            os.mkdir(self.loss_path)
        loss_path = os.path.join(self.loss_path, 'loss.json')
        with open(loss_path, 'w') as f:
            json.dump(self.loss, f) 