import torch
import argparse
import os
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.ssd import SSD
from dataset.voc import VOCDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


def set_seeds():
    SEED_VALUE = 42
    
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.cuda.manual_seed(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using mps")
    
    
def collate_fn(data):
    return tuple(zip(*data))


def train(args):
    # config/voca.yaml
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
            
    set_seeds()
    
    voc = VOCDataset('train', im_sets=dataset_config['train_im_sets'], im_size=dataset_config['im_size'])
    train_loader = DataLoader(
        dataset=voc,
        batch_size=train_config['batch_size'],
        shuffle = True,
        collate_fn=collate_fn
    )
    
    # Model Initialization
    model = SSD(config = config['model_params'], num_classes = dataset_config['num_classes'])
    model.to(device)
    model.train() # set to train mode
    
    # Load any weights if any
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print("Loading an existing checkpoint")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config['task_name'])
    
    
    # Original paper uses SGD with momentum
    optimizer = torch.optim.SGD(model.parameters(), lr = train_config["lr"],
                                weight_decay=5e-4, momentum=0.9)
    
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    
    # Store epoch-wise loss for plotting
    epoch_cls_losses = []
    epoch_loc_losses = []
    
    for epoch in range(num_epochs):
        ssd_classification_losses = []
        ssd_localization_losses = []
        
        progress_bar = tqdm(train_loader, desc = f"Epoch {epoch+1}/ {num_epochs}", leave = True)
        
        
        # Step wise loss calculation and weight update
        
        for idx, (images, targets, _) in enumerate(progress_bar):
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device) # converts to torch.int64
                
            images = torch.stack([im.float().to(device) for im in images], dim = 0)
            batch_losses, _ = model(images, targets)
            
            cls_loss = batch_losses['classification']
            loc_loss = batch_losses['bbox_regression']
            total_loss = (cls_loss + loc_loss) / acc_steps
            total_loss.backward()
            
            ssd_classification_losses.append(cls_loss)
            ssd_localization_losses.append(loc_loss)
            
            # Update weights every accumulation steps defined 
            # to simulate larger batch size training and smooth convergence
            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            progress_bar.set_postfix({
                'Cls Loss': f"{np.mean(ssd_classification_losses):.4f}",
                'Loc Loss': f"{np.mean(ssd_localization_losses):.4f}"
            })
            
            if steps % train_config['log_steps'] == 0:
                tqdm.write(
                    f"[Step {steps}] ClsLosses: {np.mean(ssd_classification_losses):.4f} | "
                    f"Loc Loss: {np.mean(ssd_localization_losses):.4f}"
                )

            if torch.isnan(total_loss):
                print("Loss is becoming NaN. Exiting...")
                exit(0)
                
            steps += 1
            
        # Epoch wise 
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        
        # Current epoch
        epoch_cls_loss = np.mean(ssd_classification_losses)
        epoch_loc_loss = np.mean(ssd_localization_losses)
        
        # All epoch
        epoch_cls_losses.append(epoch_cls_loss)
        epoch_loc_losses.append(epoch_loc_loss)
        
        print(f"\nFinished epoch {epoch + 1}")
        print*f"SSD Classification Loss: {epoch_cls_loss:.4f} | SSD Localization Loss: {epoch_loc_loss:.4f}"
        
        torch.save(model.state_dict(), ckpt_path)
    
    print("Done Training ✅")
    
    # Plot training loss graphs
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_cls_losses, label = "Classification Loss")
    plt.plot(range(1, num_epochs+1), epoch_loc_losses, label = "Localization Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(train_config['task_name'], 'training_loss.png'))
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for SSD training")
    parser.add_argument('--config', dest='config_path', default='config/voc.yaml', type = str)
    args = parser.parse_args()
    train(args)
                
    
    
    
    



