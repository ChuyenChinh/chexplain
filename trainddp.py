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

labels_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

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
                 best_model_pth,
                 u_policy
                ):
        self.rank = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.rank)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.num_labels = num_labels
        self.snap_shots_path = snap_shots_path.replace('.pt',f"_{u_policy}.pt")
        self.start_epoch = 0
        self.best_model_pth = best_model_pth.replace('.pt',f"_{u_policy}.pt")
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
        auroc_score = MultilabelAUROC(num_labels=self.num_labels,average=None).to(self.rank) #DDP automatically synchronizes AUC amongst GPUs
        #early stopping
        patience = 4
        epochs_no_improve = 0
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
                for inputs,labels in self.dataloader[phase]:
                    inputs = inputs.to(self.rank)
                    labels = labels.to(self.rank)
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

                loss_tensor = torch.tensor([running_loss,local_samples],dtype=torch.float64,device=self.rank)
                dist.all_reduce(loss_tensor)
                
                epoch_loss = (loss_tensor[0] / loss_tensor[1]).item()
                if self.rank == 0:
                    print(f"Phase {phase}: Epoch Loss = {epoch_loss}")
                
                if phase == 'train':
                    self.lr_scheduler.step()
                else:
                    epoch_auc = auroc_score.compute()
                    macro_auc = epoch_auc.mean().item()
                    if self.rank == 0:
                        for i, name in enumerate(labels_name):
                            print(f"- {name}: {epoch_auc[i].item():.4f}")
                        print(f"  Macro AUC = {macro_auc:.4f}")
                    auroc_score.reset()
                    
                    if macro_auc > self.best_auc:
                        self.best_auc = macro_auc
                        if self.rank == 0:
                            self.save_best_model()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        
                    if self.rank == 0:
                        self.save_snapshot(epoch)
                    if epochs_no_improve >= patience:
                        break
            if epochs_no_improve >= patience:
                if self.rank == 0:
                    print(f"Early stopping at epoch{epoch+1}")
                break  
                
        if self.rank == 0:
            print(f"Best auc score = {self.best_auc}")


def main(num_epochs,num_labels,u_policy):
    setup()
    model = models.densenet121(weights='IMAGENET1K_V1')
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(in_features, num_labels))
    
    dataloader = get_dataloader(u_policy) 
    optimizer = optim.AdamW(model.parameters(),weight_decay=1e-2,lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
    criterion = MaskedBCELogitLoss()
    
    trainer = Trainer(model, dataloader, optimizer, lr_scheduler, criterion, num_labels, snap_shots_path='snapshots.pt',best_model_pth='best.pt',u_policy=u_policy)
    trainer.train(num_epochs)
    
    cleanup()
    

if __name__ == '__main__':
    import sys
    num_epochs = int(sys.argv[1])
    num_labels = int(sys.argv[2])
    u_policy = sys.argv[3]
    main(num_epochs,num_labels,u_policy)
        



