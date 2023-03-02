
from torch.optim.lr_scheduler import LambdaLR


class LearningScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
    
    def reciprocal_descent(self, max_epoch):  
        reciprocal_descent_scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda= lambda x: (max_epoch - x)/max_epoch,
                verbose = False
            )
        return reciprocal_descent_scheduler
    
