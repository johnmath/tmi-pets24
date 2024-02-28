# TMI! Finetuned Models Leak Private Information from their Pretraining Data

**Authors: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman**


This repository contains the code for our PETS'24 paper, [TMI! Finetuned Models Leak Private Information from their Pretraining Data](https://arxiv.org/abs/2306.01181). The scripts in this repository will reproduce core results from the paper, but underlying library can be used to run experiments not included in the scripts.

## Training Shadow Models
The first script, `train_shadow_models.py` pretrains shadow models on the CIFAR-100 dataset and finetunes them on either CIFAR-10 or a coarse-labeled version of CIFAR-100. This script has the following arguments:

```
options:
  -h, --help            show this help message and exit
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        suffix for save directory
  -n NUM_SHADOW_MODELS, --num_shadow_models NUM_SHADOW_MODELS
                        number of shadow models
  -data DATASET, --dataset DATASET
                        finetuning dataset: {'c10': CIFAR-10, 'coarse': Coarse CIFAR-100}
  -pt PRETRAINING_EPOCHS, --pretraining_epochs PRETRAINING_EPOCHS
                        number of pretraining epochs
  -ft FINETUNING_EPOCHS, --finetuning_epochs FINETUNING_EPOCHS
                        number of finetuning epochs
  -size FINETUNING_SET_SIZE, --finetuning_set_size FINETUNING_SET_SIZE
                        number of finetuning samples
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -d DEVICE, --device DEVICE
                        PyTorch device (default: cuda:0)
  -r DATA_DIR, --data_dir DATA_DIR
                        directory for datasets
```

Suppose we would like to run this script with 64 shadow models, each pretrained for 100 epochs on CIFAR-100 and finetuned for 35 epochs on CIFAR-10. Then you would use the following command:

```shell
python train_shadow_models.py -n 64 -pt 100 -ft 35 -data c10
```

When shadow models are trained, they go to the `ShadowModels` directory and can be identified by their `experiment_name`. By default, `experiment_name` is set to `tmi`.

## Running TransferMI

Similar to training shadow models, we provide a script, `run_attacks.py` to run our attack and generate figures. This script has the following arguments:

```
  -h, --help            show this help message and exit
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        suffix for save directory
  -data DATASET, --dataset DATASET
                        finetuning dataset: {'c10': CIFAR-10, 'coarse': Coarse CIFAR-100}
  -v NUM_MODELS_TO_ATTACK, --num_models_to_attack NUM_MODELS_TO_ATTACK
                        number of models to run attack on
  -a NUM_AUGMENTATIONS, --num_augmentations NUM_AUGMENTATIONS
                        number of augmentations per image
  -d DEVICE, --device DEVICE
                        PyTorch device (default: cuda:0)
  -l USE_PRECOMPUTED_LOGITS, --use_precomputed_logits USE_PRECOMPUTED_LOGITS
                        if True, uses logits from an earlier attack. If the attack has been run once, set this to
                        True to save computation time
```

Given that 64 (+ 1) shadow models have already been trained using the command in the previous example, we can run TransferMI on 32 victim models with the following example.

```shell
python run_attacks.py -v 32 -a 8 -data c10
```

In this example, each shadow model is queried on 8 augmentations (random flips and crops) of each image. This produces 64 x 8 prediction vectors for each example. Running this command will create a directory named `Figures/` where the ROC curve plots will be saved. Because the ROC curves are plotted on a log-log scale, setting the number of victim models higher will yield smoother ROC curves as there are more membership-inference samples.