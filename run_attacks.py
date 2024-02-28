import os
import argparse
from utils.attacks import *
from utils.transfermi import TransferMI
from utils.transfermi import plot_rocs
from tqdm import tqdm
import random
from torchvision.models import resnet34

parser = argparse.ArgumentParser(description='Train Shadow Models for TransferMI Attack')

parser.add_argument("-e",
                    "--experiment_name",  
                    type=str, 
                    default="tmi",
                    help="suffix for save directory")


parser.add_argument("-data",
                    "--dataset",  
                    type=str, 
                    default="c10",
                    help="finetuning dataset: {\'c10\': CIFAR-10, 'coarse': Coarse CIFAR-100}")

parser.add_argument("-v",
                    "--num_models_to_attack",  
                    type=int, 
                    default=32,
                    help="number of models to run attack on")

parser.add_argument("-a",
                    "--num_augmentations",  
                    type=int, 
                    default=8,
                    help="number of augmentations per image")

parser.add_argument("-d",
                    "--device",  
                    type=str, 
                    default="cuda:0",
                    help="PyTorch device")


parser.add_argument("-l",
                    "--use_precomputed_logits",  
                    type=bool, 
                    default=False,
                    help="if True, uses logits from an earlier attack. If the attack has been run once, set this to True to save computation time")


args = parser.parse_args()

experiment_name = args.experiment_name
device = args.device
dataset_choice = args.dataset
num_models_to_attack = args.num_models_to_attack
num_augmentations = args.num_augmentations

# Construct attack dictionary
attacks = AttackFactory(
    TMI,
    AdaptedOnlineLira,
)


total_outs = {
    attack_name: torch.Tensor([])
    for attack_name in attacks.keys()
}

total_gts = torch.Tensor([])

# Pick dataset to initialize TransferMI object
if dataset_choice == "c10":
    num_classes = 10
    ft_task = "CIFAR-10"
elif dataset_choice == "coarse":
    num_classes=20
    ft_task = "Coarse CIFAR-100"

save_dir = "CIFAR100"
first_done=args.use_precomputed_logits
dp_param=None

tmi_obj = TransferMI(
    experiment_name=experiment_name, 
    model=resnet34(num_classes=num_classes), 
    device=device,
    load_saved=True,
    manual_save_dir=save_dir,
)

model_names = list(filter(lambda x : "transfer" in x, os.listdir(tmi_obj.saved_models_dir)))
plot_title_string = f"Original Task: CIFAR-100 \n Fine-Tune Task: {ft_task}"

# Run attacks on the chosen number of models
# Each round, one is randomly chosen as the 
# victim and the remaining N are designated as shadow models
for model_index in tqdm(random.sample(range(len(model_names)), num_models_to_attack), desc="Attacking Models using Mask Matrix"):   
    
    scores, ground_truths = tmi_obj.run_attacks(model_index, 
                                                attacks, 
                                                num_augmentations=num_augmentations, 
                                                dp_param=dp_param,
                                                use_saved_logits=first_done)
    first_done=True
    
    for attack_name, mis in scores.items():
        total_outs[attack_name] = torch.concatenate((total_outs[attack_name], mis))
    
    total_gts = torch.concatenate((total_gts, ground_truths))

    
if not os.path.exists(f"Figures"):
    os.mkdir("Figures/")
    os.mkdir(f"Figures/{save_dir}")
elif not os.path.exists(f"Figures/{save_dir}"):
    os.mkdir(f"Figures/{save_dir}")

# Plots ROC curves
# If attack is run on few shadow models, the plot
# may look jagged. For smoother plots on the log-log 
# scale, run attacks on more models
_, _ = plot_rocs(total_outs, 
                 total_gts, 
                 pos_label=1, 
                 save_file=True,
                 title=plot_title_string, 
                 file_name=f"{save_dir}/{experiment_name}")
