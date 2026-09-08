import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models   
from feature import get_dataloader
from loss_criterion import MaskedBCELogitLoss
import psutil

def print_mem(tag=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024**2  # MB
    print(f"[{tag}] RAM used: {rss:.2f} MB")

def setup():
    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)

    dist.init_process_group(backend)
    
def cleanup():
    dist.destroy_process_group()

class Trainer:
    def __init__(self,
                 model,
                 dataloader,
                 optimizer,
                 lr_scheduler,
                 criterion,
                 num_labels,
                 snap_shots_path,
                 best_model_pth
                ):
        self.rank = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.rank)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.num_labels = num_labels
        self.snap_shots_path = snap_shots_path
        self.start_epoch = 0
        self.best_model_pth = best_model_pth
        self.best_auc = 0.0
        if os.path.exists(snap_shots_path):
            self.load_snapshot(snap_shots_path)
        self.model = DDP(self.model,device_ids=[self.rank])
        
    def load_snapshot(self,pth):
        device_type = torch.accelerator.current_accelerator().type
        loc = f"{device_type}:{self.rank}"
        snapshot = torch.load(pth,map_location=loc)
        
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.optimizer.load_state_dict(snapshot['OPTIMIZER_STATE'])
        self.lr_scheduler.load_state_dict(snapshot['LR_SCHED_STATE'])
        self.start_epoch = snapshot['EPOCHS_RUN']
        self.best_auc = snapshot.get('BEST_AUC',0.0)
        
    def save_snapshot(self,epoch):
        snapshots = {}
        snapshots['MODEL_STATE'] = self.model.module.state_dict()
        snapshots['OPTIMIZER_STATE'] = self.optimizer.state_dict()
        snapshots['LR_SCHED_STATE'] = self.lr_scheduler.state_dict()
        snapshots['EPOCHS_RUN'] = epoch + 1
        snapshots['BEST_AUC'] = self.best_auc
        torch.save(snapshots,self.snap_shots_path)
        
    def save_best_model(self):
        torch.save(self.model.module.state_dict(),self.best_model_pth)
        
    def train(self,num_epochs):
        auroc_score = MultilabelAUROC(num_labels=self.num_labels).to(self.rank) #DDP automatically synchronizes AUC amongst GPUs

        for epoch in range(self.start_epoch,num_epochs):
            if self.rank == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("-" * 20)
            
            self.dataloader['train'].sampler.set_epoch(epoch)
            
            for phase in ['train','valid']:
                if phase == 'train':
                    self.model.train()
                elif phase == 'valid':
                    self.model.eval()
                running_loss = 0.0
                local_samples = 0
                for batch_idx , (inputs,labels) in enumerate(self.dataloader[phase]):
                    inputs = inputs.to(self.rank,non_blocking=True)
                    labels = labels.to(self.rank,non_blocking=True)
                    
                    with torch.set_grad_enabled(phase=='train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits,labels)
                        
                        if phase == 'train':
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            self.optimizer.step()
                        elif phase == 'valid':
                            probs = torch.sigmoid(logits)
                            auroc_score.update(probs.detach(),labels.detach().int())

                    running_loss += loss.detach().item() * inputs.shape[0]
                    local_samples += inputs.shape[0]
                    if self.rank == 0 and batch_idx % 50 == 0:
                        print_mem()

                loss_tensor = torch.tensor([running_loss,local_samples],dtype=torch.float64,device=self.rank)
                dist.all_reduce(loss_tensor)
                
                epoch_loss = (loss_tensor[0] / loss_tensor[1]).item()
                if self.rank == 0:
                    print(f"Phase {phase}: Epoch Loss = {epoch_loss}")
                
                if phase == 'train':
                    self.lr_scheduler.step()
                else:
                    epoch_auc = auroc_score.compute().item()
                    if self.rank == 0:
                        print(f"Phase {phase}: Macro AUC = {epoch_auc:.4f}")
                    auroc_score.reset()
                
                if self.rank == 0 and phase == 'valid':
                    if epoch_auc > self.best_auc:
                        self.best_auc = epoch_auc
                        self.save_best_model()
                    self.save_snapshot(epoch)
        if self.rank == 0:
            print(f"Best auc score = {self.best_auc}")


def main(num_epochs,num_labels):
    setup()
    model = models.densenet121(weights='IMAGENET1K_V1')
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    
    dataloader = get_dataloader() 
    optimizer = optim.AdamW(model.parameters(),weight_decay=1e-4,lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
    criterion = MaskedBCELogitLoss()
    
    trainer = Trainer(model, dataloader, optimizer, lr_scheduler, criterion, num_labels, snap_shots_path='snapshots.pt',best_model_pth='best.pt')
    trainer.train(num_epochs)
    
    cleanup()
    

if __name__ == '__main__':
    import sys
    num_epochs = int(sys.argv[1])
    num_labels = int(sys.argv[2])
    main(num_epochs,num_labels)
        



