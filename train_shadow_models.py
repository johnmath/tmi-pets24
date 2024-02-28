import random
import warnings
from tqdm import tqdm
import numpy as np
import argparse
from utils.data import CoarseCIFAR100
from utils.transfermi import TransferMI

import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet34, resnet101
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train Shadow Models for TransferMI Attack')

parser.add_argument("-e",
                    "--experiment_name",  
                    type=str, 
                    default="tmi",
                    help="suffix for save directory")

parser.add_argument("-n",
                    "--num_shadow_models",  
                    type=int, 
                    default=64,
                    help="number of shadow models")

parser.add_argument("-data",
                    "--dataset",  
                    type=str, 
                    default="c10",
                    help="finetuning dataset: {\'c10\': CIFAR-10, 'coarse': Coarse CIFAR-100}")

parser.add_argument("-pt",
                    "--pretraining_epochs",  
                    type=int, 
                    default=100,
                    help="number of pretraining epochs")

parser.add_argument("-ft",
                    "--finetuning_epochs",  
                    type=int, 
                    default=35,
                    help="number of finetuning epochs")

parser.add_argument("-size",
                    "--finetuning_set_size",  
                    type=int, 
                    default=None,
                    help="number of finetuning samples")

parser.add_argument("-lr",
                    "--learning_rate",  
                    type=float, 
                    default=0.01,
                    help="learning rate")

parser.add_argument("-b",
                    "--batch_size",  
                    type=int, 
                    default=128,
                    help="batch size")

parser.add_argument("-d",
                    "--device",  
                    type=str, 
                    default="cuda:0",
                    help="PyTorch device")

parser.add_argument("-r",
                    "--data_dir",  
                    type=str, 
                    default="data",
                    help="directory for datasets")

args = parser.parse_args()

transforms = T.Compose([T.ToTensor(), 
                        T.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                        T.RandomHorizontalFlip(0.5), 
                        T.RandomCrop(size=(32,32), 
                                     padding=4, 
                                     padding_mode="reflect")],
                      )



# Set up variables for pretraining
save_dir = "CIFAR100"
root = args.data_dir
experiment_name = args.experiment_name
num_classes = 100
dp_param = None
manual_save_name_logits = ""
manual_save_name_model = ""
num_shadow_models = args.num_shadow_models
pretraining_epochs = args.pretraining_epochs
load_saved = False
learning_rate = args.learning_rate
batch_size = args.batch_size
device = args.device
pretraining_dataset = CIFAR100(root=root, download=True, train=True, transform=transforms)


# Pretrain (num_shadow_models + 1) shadow models on CIFAR-100
# This object keeps track of which points were used to train 
# which shadow model, along with many other training utils
tmi_obj = TransferMI(
    experiment_name=experiment_name, 
    model=resnet34(num_classes=num_classes), 
    output_dim=num_classes, 
    dataset=pretraining_dataset, 
    criterion=nn.CrossEntropyLoss(), 
    train_size=len(pretraining_dataset)//2, 
    holdout_size=0, 
    num_shadow_models=num_shadow_models,
    epochs=pretraining_epochs, 
    lr=learning_rate, 
    batch_size=batch_size,
    manual_save_dir=save_dir,
    load_saved=load_saved,
    device=device
)

# Fine-tune all shadow models
dataset_choice = args.dataset

if dataset_choice == "c10":
    original_transfer_set = CIFAR10(root=root, download=True, train=True, transform=transforms)
    num_transfer_classes = 10
elif dataset_choice == "coarse":
    original_transfer_set = CoarseCIFAR100(root=root, train=False, transform=transforms)
    num_transfer_classes = 20


finetune_dataset_size = args.finetuning_set_size
finetuning_epochs = args.finetuning_epochs
finetuning_learning_rate = args.learning_rate

# Finetune each of the N+1 pretrained
# shadow models
for i in range(num_shadow_models+1):  
    
    # Take uniform random samples of the finetuning data
    if finetune_dataset_size:
        sample = random.sample(range(0,len(original_transfer_set)), k=finetune_dataset_size)
    else:
        sample = random.sample(range(0,len(original_transfer_set)), k=int(0.85*len(original_transfer_set)))
        
    transfer_dataset = Subset(original_transfer_set, sample)
    
    _ = tmi_obj.standard_transfer(
        model_ind=i, 
        transfer_set=transfer_dataset, 
        num_classes=num_transfer_classes, 
        epochs=finetuning_epochs,
        lr=finetuning_learning_rate, 
        save=True, 
        scheduler=True, 
        scheduler_step=10,
        named_layers=["fc"],
    )
