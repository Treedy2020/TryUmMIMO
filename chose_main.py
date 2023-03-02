import yaml
import argparse
import torch.optim as optim
from torch.nn.functional import mse_loss
from utils.dataloader import DataGenerator, MimoDataset, MimoDataLoader
from utils.model import FeatureExtractModel, ExtractionStrategies, ChoseNet 
from utils.scheduler import LearningScheduler
from utils.trainer import Trainer


def train(config):
    # Get dataloader
    data_generator = DataGenerator(Nt=config['Nt'],
                             Nr=config['Nr'],
                             c=config['c'],
                             c_sub=config['c_sub'],
                             number=config['number'],
                             max_value=config['max_value'],
                             data_root=config['data_root'])
    raw_data = data_generator.generate()
    mimo_dataset = MimoDataset(raw_data_array=raw_data)
    mimo_dataloader = MimoDataLoader(dataset=mimo_dataset, batch_size= config['batch_size'])
    train_loader, val_loader, test_loader = mimo_dataloader.getDataLoader()
    
    # Get Extract Model
    f = FeatureExtractModel(config['input_dim'])
    restore_model = f.getLinear()

    # Get ExtractStrategies
    input_dim = config['input_dim']
    es = ExtractionStrategies(input_dim, compress_ratio=config['compress_ratio'])
    # self.c_sub*self.c, self.Nt*self.Nr
    chose_net = ChoseNet(es.equidistantExtraction(), restore_model=restore_model, input_num=int(config['c_sub']*config['c']//config['compress_ratio']), restore_num=config['c_sub']*config['c'])
    
    # Get Trainer
    criterion = mse_loss
    optimizer = optim.Adam(params=chose_net.parameters(), lr=1e-3)
    max_epoch = config['epoch']
    
    # Get Scheduler
    sch = LearningScheduler(optimizer)
    scheduler = sch.reciprocal_descent(max_epoch)
    
    
    trainer = Trainer(model=chose_net,
                      epoch=config['epoch'],
                      train_dataloader=train_loader,
                      val_dataloader=val_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      lr_scheduler=scheduler,
                      checkpoint_path= config['checkpoint_path'],
                      loss_path=config['loss_path'],
                      )
    trainer.train()


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('--config', type=str, help='Path of  config.yaml')
    args = paser.parse_args()
    
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)