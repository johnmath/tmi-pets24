import os
import copy
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .models import ModelUtility
from .models import NeuralNet
from .attacks import *

def sweep(score, y, pos_label):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(y, score, pos_label=pos_label)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    return fpr, tpr, auc(fpr, tpr), acc


def plot_rocs(scores, ground_truth, pos_label=1, title="", save_file=False, file_name=None):
    """
    Plot the PR Curves with log scaling on both axes
    """
    legend_str = " "
    x = np.linspace(0, 1, 100)
    y = x
    
    misc_scores = {}
    attacks_dict = {}
    plots_dict = {}
    for attack, to_score in scores.items():
        fpr, tpr, auc, acc = sweep(to_score, ground_truth, pos_label)
        plots_dict[attack] = {"fpr": fpr, "tpr": tpr}
        print(f"{attack}:\nAccuracy: {acc}\nAUC: {auc}\n*******************************\n")
        plt.title(title)
        plt.plot(fpr, tpr, label=attack + f", auc={auc:.3f}", linewidth=2)
        
        misc_scores[attack] = {
            "Accuracy": acc,
            "AUC": auc,
        }
    
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-4,1)
    plt.ylim(1e-4,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.legend(loc="lower right")

    if save_file:
        
        if not os.path.exists("Figures"):
            os.mkdir("Figures")
        
        if file_name:
            plt.savefig(f"Figures/{file_name}.pdf")
            np.save(f"Figures/{file_name}_scores.npy", misc_scores)
            np.save(f"Figures/{file_name}_plotdata.npy", plots_dict)
        else:
            raise ValueError("If save_file=True, file_name cannot be None")
    else:
        plt.show()
        
    return x, y


def reset_weights(model):
    """
    Reset the weights of provided model in place.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

class TransferMI:
    def _evaluate_accuracy(self, model, data_loader):
        total_correct = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
        return total_correct / len(data_loader.dataset)
        
    def _train_shadow_model(
        self,
        train_loader,
        shadow_model_number,
        optimizer=torch.optim.SGD,
    ):
        """Helper function to train individual shadow models
        Parameters
        ----------
            random_sample : PyTorch Dataloader
                The randomly generated dataset to train a single shadow model
            shadow_model_number : int
                Which shadow model we are training
            gamma : float
                Multiplier for learning rate scheduler
            scheduler_step : int
                Number of epochs before multiplying learning rate by gamma
            optimizer : torch.Optimizer
                Optimizer for torch training
        """
        shadow_model = copy.deepcopy(self.model).to(self.device)
        reset_weights(shadow_model)

        optimizer = self.optimizer(
            shadow_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4
        )
        
        model_wrap = ModelUtility(
            model=shadow_model,
            criterion=self.criterion,
            optimizer=None,
            override_optimizer=optimizer,
            lr=0.01,
            scheduler=True,
            schedule_step=None,
            gamma=None,
            out_features=self.output_dim,
            device=self.device,
            prefix=self.saved_models_dir,
        )
        
        dataloaders = {"train": train_loader}

        shadow_model, _ = model_wrap.standard_fit(
            dataloaders=dataloaders,
            num_epochs=self.epochs,
            start_epoch=0,
            save=False,
            train_only=True,
            desc_string=f"Pretraining Shadow Model {shadow_model_number}"
        )
                
#         shadow_model.train()
#         for _ in tqdm(range(self.epochs), desc=f"Training Shadow Model {shadow_model_number}"):
#             running_loss = 0
#             for (inputs, labels) in random_sample:

#                 optimizer.zero_grad()

#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 outputs = shadow_model(inputs)

#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#             # scheduler.step()

#         print(
#             f"Shadow Model Final Training Error: {running_loss/len(random_sample.dataset):.4}\n"
#             + f"Shadow Model Final Training Accuracy: {self._evaluate_accuracy(shadow_model, random_sample)*100:.5}%"
#         )
#         print("-" * 8)

        shadow_model.eval()
        torch.save(
            shadow_model,
            f"{self.saved_models_dir}/shadow_model_{shadow_model_number}",
        )
        
    def _train_models(self):
        print(f"The total dataset size is: {self.sampled_distribution_size}")
        print(f"Each shadow model will be trained on: {self.train_size} samples")
        print(f"Each attack will use: {self.num_target_points} TARGET points")
        self.target_indices = self.random.choice(
            list(range(self.sampled_distribution_size)),
            self.num_target_points,
            replace=False,
        )
        self.shadow_indices = [
            self.random.choice(
                list(range(self.sampled_distribution_size)),
                self.train_size,
                replace=False,
            )
            for _ in range(self.num_shadow_models)
        ]

        # Mask reference for in/out points in each shadow model training
        self.mask = np.ndarray(shape=(self.num_target_points, self.num_shadow_models))
        for i, ind in enumerate(self.target_indices):
            for shadow_model in range(self.num_shadow_models):
                # Decide whether or not it is in shadow model training
                self.mask[i][shadow_model] = (
                    1 if ind in self.shadow_indices[shadow_model] else 0
                )

        for shadow_model in range(self.num_shadow_models):
            training_subset = torch.utils.data.Subset(
                self.dataset, self.shadow_indices[shadow_model]
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=training_subset, batch_size=self.batch_size, shuffle=True, num_workers=16, persistent_workers=True
            )

            self._train_shadow_model(
                train_loader,
                shadow_model_number=shadow_model + 1,
                optimizer=self.optimizer,
            )

        entries = copy.deepcopy(vars(self))
        entries.pop("load_saved")
        np.save(f"{self.saved_models_dir}/{self.name}.npy", entries, allow_pickle=True)

    def __init__(
        self,
        experiment_name,
        device,
        model,
        output_dim=None,
        dataset=None,
        criterion=None,
        train_size=None,
        holdout_size=None,
        epochs=None,
        lr=None,
        batch_size=None,
        optimizer=torch.optim.SGD,
        momentum=0.9,
        num_shadow_models=50,
        num_target_points=1000,
        load_saved=False,
        seed=0,
        manual_save_dir=None,
    ):
        """Utility for performing membership inference attacks on deep
        learning models that have been transferred from another task
            ...
            Attributes
            ----------
            experiment_name : str
                Name for this experiment
            model : PyTorch model
                A model architecture to use for TransferMI attacks
            output_dim : int
                Dimensions of the output data
            dataset : PyTorch Dataset
                A dataset contained WHOLE distribution of points (which will be partitioned)
            criterion: PyTorch criterion
                An criterion to use for this model
            train_size: int
                Size of training sets
            holdout_size: int
                How many points to holdout for updating
            epochs : int
                Number of epochs for training
            lr : float
                Training learning rate
            batch_size : int
                Training batch size
            optimizer: PyTorch optimizer
                An optimizer to use for this model
            momentum: float
                Momentum parameter for optimzer
            num_shadow_models : int
                The number of shadow models to use in the attack
            num_target_points : int
                The number of points to use in membership inference attacks
            load_saved : bool
                Whether to load saved models or train shadow from scratch
            device : int
                The device for Pytorch to map to
            seed : int
                The random seed for partitioning
        """
        
        
        if load_saved:
            self.name = experiment_name
            self.model = model
            if not manual_save_dir:
                self.save_dir = str(type(model)).split(".")[-1].split("'")[0]
            else:
                self.save_dir = manual_save_dir
            self.saved_models_dir = f"ShadowModels/{self.save_dir}_{self.name}"
            
            attributes = np.atleast_1d(np.load(f"ShadowModels/{self.save_dir}_{self.name}/{self.name}.npy", allow_pickle=True))[0]
            self.__dict__.update(attributes)
            self.device=device
        else:
            self.name = experiment_name
            self.model = model
            self.output_dim = output_dim
            self.criterion = criterion
            self.train_size = train_size
            self.holdout_size = holdout_size
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.optimizer = optimizer
            self.momentum = momentum
            self.num_shadow_models = num_shadow_models + 1
            self.num_target_points = num_target_points
            self.load_saved = load_saved
            self.device = device
            self.random = np.random.RandomState(seed)
            torch.manual_seed(seed)
            if not manual_save_dir:
                self.save_dir = str(type(model)).split(".")[-1].split("'")[0]
            else:
                self.save_dir = manual_save_dir
            self.saved_models_dir = f"ShadowModels/{self.save_dir}_{self.name}"
            if not os.path.exists("ShadowModels"):
                os.mkdir("ShadowModels")

            if not os.path.exists(self.saved_models_dir):
                os.mkdir(self.saved_models_dir)

            self.total_distribution_size = len(dataset)
            self.sampled_distribution_size = (
                self.total_distribution_size - self.holdout_size
            )
            self.dataset, self.holdout = torch.utils.data.random_split(
                dataset,
                [self.sampled_distribution_size, self.holdout_size],
            )
            self._train_models()
        

        
    def change_transforms(self, new_transforms):
        self.dataset.dataset.transform = new_transforms

    def load_from_saved(location, device=None):
        entries = np.load(location, allow_pickle=True).item()
        if device:
            entries[device] = device

        return TransferMI(experiment_name=entries["name"], load_saved=True, **entries)

    def _logit_scaling(self, p):
        """Perform logit scaling so that the model's confidence is
        approximately normally distributed

        Parameters
        ----------
            p : torch.Tensor
                A tensor containing some model's confidence scores

        Returns
        -------
            phi(p) : PyTorch.Tensor(float)
                The scaled model confidences
        """
        assert isinstance(p, torch.Tensor)
        # p = torch.clamp(p, min=10e-16, max=1-10e-16)
        # for stability purposes
        return torch.log(p+10e-16) - torch.log((1-p)+10e-16)
        
#  Computationally Stable Rescaling
#         masks = np.eye(p.shape[1]).astype(bool)
#         computed_logits = []
#         print(p.shape)
#         for i, mask in enumerate(masks):
#             idx = p*torch.Tensor(mask).to(p.device)
#             not_idx = p*torch.Tensor(~mask).to(p.device)
#             # Logit Scaling
#             prob = idx.sum(dim=1)
#             one_minus_prob = not_idx.sum(dim=1)
#             scaled = torch.log(prob) - torch.log(one_minus_prob)
#             computed_logits.append(scaled)
#         return torch.stack(computed_logits).T

    def _model_confidence(
        self,
        model, 
        dataloader,
        device,
    ):
        """Helper function to calculate the model confidence on provided examples

        Model confidence is defined as softmax probability of the highest probability class

        Parameters
        ----------
            model : PyTorch model
                A Pytorch machine learning model
            datapoints : torch.Dataloader
                Dataloader to get confidence scores for
            device : str

        Returns
        -------
            model_confidence : List(float)
                softmax(model(x_n)) on the y_n class for nth datapoint
        """
        model.eval()
        softmax = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                x = x.to(self.device)
                predictions = model(x)

        softmax_values = softmax(predictions)
        
        return softmax_values
    
    def _raw_logits(
        self,
        model, 
        dataloader,
        device,
    ):
        """Helper function to query a model on a dataset

        Parameters
        ----------
            model : PyTorch model
                A Pytorch machine learning model
            datapoints : torch.Dataloader
                Dataloader to get confidence scores for
            device : str

        Returns
        -------
            predictions : List(float)
                model(x_n) (i.e. the raw output logits before a softmax is applied)
        """
        model.eval()

        with torch.no_grad():
            model = model.to(device)
            
            # Run in one batch to speed up calculations
            for x, y in dataloader:
                x = x.to(self.device)
                predictions = model(x)

        return predictions
    
    def _raw_augmented_logits(
        self,
        model, 
        dataloader,
        num_augmentations,
        device,
        outer_batch_size=16,
    ):
        """Helper function to query a model on a dataset with several augmentations

        Parameters
        ----------
            model : PyTorch model
                A Pytorch machine learning model
            datapoints : torch.Dataloader
                Dataloader to get confidence scores for
            num_augmentations : int
                The number of augmentations to query over
            device : str


        Returns
        -------
            predictions : List(float)
                model(x_n) (i.e. the raw output logits before a softmax is applied)
        """
        model.eval()
        preds_each = [torch.Tensor([]).to(device) for _ in range(self.num_target_points)]
        
        with torch.no_grad():
            model = model.to(device)
            
            for aug in range(num_augmentations//outer_batch_size):
                outer = torch.Tensor([]).to(device)
                for _ in range(outer_batch_size):
                    for x, _ in dataloader:
                        outer = torch.concat([outer, x.to(device)])
                    
                outs = model(outer)
                for idx in range(len(outs)//outer_batch_size):
                    preds_each[idx] = torch.concat([preds_each[idx], outs[[idx + (self.num_target_points*j) for j in range(outer_batch_size)]]], dim=0)
        
        return torch.stack(preds_each).cpu().flatten(0,1)
            
    
    def run_attacks(self, target_model_ind: int, attacks: dict, dp_param=None, num_augmentations=None, whitebox=False, use_saved_logits=False, manual_save_name_model="", manual_save_name_logits="", k=None):
        """
        Modular method to run different kinds of MI attacks
        This attack performs membership inference on the pretraining set of a transfer learning model
        
        Parameters
        --------------
            target_model_ind : int
                The index of the desired target model. 
                The target model will be one of the shadow models with path "ShadowModels/{model_experiment}/shadow_model_{target_model_int}_transfer"
            attacks : dict
                Dicitonary of attack objects to run
            dp_param : int
                If None, non-DP models are used. Else, will run attacks on models trained with DP 
                (eps=dp_param, delta=10^-5)
            num_augmentations : int
                The number of aigmentations per queried challenge point. Default value is None
            manual_save_name : str
                Suffix for situations where multiple layers were fine-tuned
        """
        
        if num_augmentations is not None and num_augmentations <= 1:
            num_augmentations = None
        
        ground_truths = []
        tmi_dataset = torch.utils.data.Subset(self.dataset, self.target_indices)
        tmi_loader = torch.utils.data.DataLoader(tmi_dataset, batch_size=self.num_target_points, shuffle=False)
        for i, ind in enumerate(self.target_indices):
            ground_truths.append(self.mask[i][target_model_ind])
        ground_truths = torch.Tensor(ground_truths)
        
        target_point_logits = [torch.Tensor() for _ in range(len(self.target_indices))]
        target_point_scaled_logits = [torch.Tensor() for _ in range(len(self.target_indices))]
        clipped_logits = [torch.Tensor() for _ in range(len(self.target_indices))]
        raw_clipped_logits = [torch.Tensor() for _ in range(len(self.target_indices))]
        if not use_saved_logits:
            for i in tqdm(
                range(self.num_shadow_models), 
                desc=f"Running Inference on Shadow Models", 
                position=0, 
                leave=True,
            ):
                
                if not whitebox:
                    load_string = f"{self.saved_models_dir}/shadow_model_{i+1}_transfer" + manual_save_name_model
                else:
                    load_string = f"{self.saved_models_dir}/shadow_model_{i+1}"
                if dp_param:
                    load_string += f"_DP_eps={dp_param}"
                shadow_model = torch.load(
                    load_string,
                    map_location=self.device,
                )
                
                shadow_model.eval()
                sm = torch.nn.Softmax(dim=1)

                if num_augmentations:
                    raw_outs = self._raw_augmented_logits(shadow_model, tmi_loader, num_augmentations, self.device, outer_batch_size=min(num_augmentations, 8)).detach().cpu()

                    raw_outs_norms = raw_outs.norm(p=2, dim=1)

                    raw_outs_norms[raw_outs_norms < 1] = 1
                    raw_clipped_outs = raw_outs/raw_outs_norms.unsqueeze(1)

                    outs = sm(raw_outs)
                    
                    if k:
                        outs = self.get_topk_logits(outs, k=k)
                    
                    scaled_outs = self._logit_scaling(outs)
                    scaled_outs_norms = scaled_outs.norm(p=2, dim=1)
                    scaled_outs_norms[scaled_outs_norms < 1] = 1
                    clipped_outs = scaled_outs/scaled_outs_norms.unsqueeze(1)

                    outs = outs.reshape((outs.shape[0]//num_augmentations, num_augmentations, outs.shape[1]))
                    scaled_outs = scaled_outs.reshape((scaled_outs.shape[0]//num_augmentations, num_augmentations, scaled_outs.shape[1]))
                    clipped_outs = clipped_outs.reshape((clipped_outs.shape[0]//num_augmentations, num_augmentations, clipped_outs.shape[1]))
                    raw_clipped_outs = raw_clipped_outs.reshape((raw_clipped_outs.shape[0]//num_augmentations, num_augmentations, raw_clipped_outs.shape[1]))

                else:
                    raw_outs = self._raw_logits(shadow_model, tmi_loader, self.device).detach().cpu()
                # Clipped logits for metaclassifier
                    raw_outs_norms = raw_outs.norm(p=2, dim=1)

                    raw_outs_norms[raw_outs_norms < 1] = 1
                    raw_clipped_outs = raw_outs/raw_outs_norms.unsqueeze(1)

                    outs = sm(raw_outs)
                    
                    if k:
                        outs = self.get_topk_logits(outs, k=k)
                    
                    scaled_outs = self._logit_scaling(outs)
                    scaled_outs_norms = scaled_outs.norm(p=2, dim=1)
                    scaled_outs_norms[scaled_outs_norms < 1] = 1
                    clipped_outs = scaled_outs/scaled_outs_norms.unsqueeze(1)
                    
                for idx in range(len(outs)):
                    target_point_logits[idx] = torch.concatenate([target_point_logits[idx], outs[idx].unsqueeze(0)], dim=0)
                    target_point_scaled_logits[idx] = torch.concatenate([target_point_scaled_logits[idx], scaled_outs[idx].unsqueeze(0)], dim=0)
                    clipped_logits[idx] = torch.concatenate([clipped_logits[idx], clipped_outs[idx].unsqueeze(0)], dim=0)
                    raw_clipped_logits[idx] = torch.concatenate([raw_clipped_logits[idx], raw_clipped_outs[idx].unsqueeze(0)], dim=0)
            

            target_point_logits = torch.stack(target_point_logits)
            target_point_scaled_logits = torch.stack(target_point_scaled_logits)
            clipped_logits = torch.stack(clipped_logits)
            # raw_clipped_logits = torch.stack(raw_clipped_logits)
            
            if not num_augmentations:
                target_point_logits = target_point_logits.unsqueeze(2)
                target_point_scaled_logits = target_point_scaled_logits.unsqueeze(2)
                clipped_logits = clipped_logits.unsqueeze(2)
                raw_clipped_logits = raw_clipped_logits.unsqueeze(2)
                
            torch.save(target_point_logits, self.saved_models_dir + f"/target_point_logits{manual_save_name_logits}.pth")
            torch.save(target_point_scaled_logits, self.saved_models_dir + f"/target_point_scaled_logits{manual_save_name_logits}.pth")
            torch.save(clipped_logits, self.saved_models_dir + f"/clipped_logits{manual_save_name_logits}.pth")
            torch.save(raw_clipped_logits, self.saved_models_dir + f"/raw_clipped_logits{manual_save_name_logits}.pth")
        else:
            target_point_logits = torch.load(self.saved_models_dir + f"/target_point_logits{manual_save_name_logits}.pth", map_location="cpu")
            target_point_scaled_logits = torch.load(self.saved_models_dir + f"/target_point_scaled_logits{manual_save_name_logits}.pth", map_location="cpu")
            clipped_logits = torch.load(self.saved_models_dir + f"/clipped_logits{manual_save_name_logits}.pth", map_location="cpu")
            raw_clipped_logits = torch.load(self.saved_models_dir + f"/raw_clipped_logits{manual_save_name_logits}.pth", map_location="cpu")
        
        # Load target model into memory 
        
        
        if not whitebox:

            target_model = torch.load(
                f"{self.saved_models_dir}/shadow_model_{target_model_ind+1}_transfer" + manual_save_name_model,
                map_location=self.device
            )
        else:
            target_model = torch.load(
                f"{self.saved_models_dir}/shadow_model_{target_model_ind+1}" + manual_save_name_model,
                map_location=self.device
            )
        target_model.eval()
        all_membership_scores = {
            attack: torch.Tensor([]) for attack in attacks.keys()
        }
        for challenge_pt in tqdm(
            range(self.num_target_points), 
            desc=f"Running Attack on {self.num_target_points} Target Points", 
            position=0, 
            leave=True
        ):
            for attack_name, attack_init in attacks.items():
                attack_object = attack_init(self.dataset, self.target_indices, self.device)
                membership_score = attack_object.run_attack(
                    target_model=target_model,
                    target_model_ind=target_model_ind, 
                    challenge_pt=challenge_pt,
                    target_point_logits=target_point_logits, 
                    target_point_scaled_logits=target_point_scaled_logits,
                    clipped_logits=clipped_logits,
                    raw_clipped_logits=raw_clipped_logits,
                    mask=self.mask,
                )
                # membership_score = torch.tensor([score])
                all_membership_scores[attack_name] = torch.concatenate(
                    [all_membership_scores[attack_name], membership_score], 
                    dim=0
                )
        
        for attack_name, unbounded_scores in all_membership_scores.items():
            # min-max norm
            if "Transfer" in attack_name:
                continue
            min_v = min([predictions.min() for predictions in unbounded_scores])
            max_v = max([predictions.max() for predictions in unbounded_scores])
            all_membership_scores[attack_name] = torch.tensor([(predictions - min_v) / (max_v - min_v) for predictions in unbounded_scores])

        return all_membership_scores, ground_truths
    
    def standard_transfer(self, 
                          model_ind, 
                          transfer_set,
                          num_classes, 
                          epochs, 
                          lr, 
                          scheduler=False, 
                          scheduler_step=10, 
                          gamma=0.9, 
                          save=True,
                          last_layer_name="fc",
                          named_layers=["fc"],
                          manual_save_name=None,
    ):
        """
        Performs transfer learning on a shadow model using a new dataset.
        
        Parameters
        ----------
            model_ind : int
                Which shadow model to update
            transfer_set : PyTorch Dataset
                The dataset to train on
            num_classes : int 
                The number of classes in the transfer learning task
            epochs : int
                How many epochs to update with
            lr : float
                The learning rate
            scheduler : bool
                If true, uses a learning rate scheduler while training
            scheduler_step: int
                Scheduling step parameter
            gamma: float
                Gamma value for scheduler
            save: bool
                Whether or not to save results of update
        """
        target_model = torch.load(
            f"{self.saved_models_dir}/shadow_model_{model_ind+1}",
            map_location=self.device,
        )
        
        target_model._modules[last_layer_name] = torch.nn.Linear(in_features=target_model._modules[last_layer_name].in_features, out_features=num_classes)
        
        for name, param in target_model.named_parameters():
            found_layer_flag = False
    
            for named_layer in named_layers:
                if named_layer in name:
                    found_layer_flag = True
            
            if not found_layer_flag:
                param.requires_grad = False
        
        optimizer = torch.optim.SGD
        model_wrap = ModelUtility(
            model=target_model,
            criterion=self.criterion,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            schedule_step=scheduler_step,
            gamma=gamma,
            out_features=self.output_dim,
            device=self.device,
            prefix=self.saved_models_dir,
        )
        
        dataloaders = {
            "train": torch.utils.data.DataLoader(
                dataset=transfer_set, batch_size=self.batch_size, shuffle=True, num_workers=16, persistent_workers=True
            )
        }

        out_model, _ = model_wrap.standard_fit(
            dataloaders=dataloaders,
            num_epochs=epochs,
            start_epoch=0,
            save=False,
            train_only=True,
            desc_string=f"Fine-tuning Model {model_ind+1}"
        )
        
        if manual_save_name:
            manual_save_name = "_" + manual_save_name
        else:
            manual_save_name = ""
        
        if save:
            torch.save(out_model, f"{self.saved_models_dir}/shadow_model_{model_ind+1}_transfer" + manual_save_name)
        return out_model
        
#     def private_transfer(
#         self, 
#         model_ind, 
#         transfer_set,
#         num_classes, 
#         epochs, 
#         lr,
#         target_epsilon=1,
#         target_delta=10e-5,
#         scheduler=False, 
#         scheduler_step=10, 
#         gamma=0.9,
#         save=True,
#         last_layer_name="fc",
#         named_layers=["fc"],
#         manual_save_name=None,
#     ):
#         """
#         Performs differentially private transfer learning on a shadow model using a new dataset.
        
#         Parameters
#         ----------
#             model_ind : int
#                 Which shadow model to update
#             transfer_set : PyTorch Dataset
#                 The dataset to train on
#             num_classes : int 
#                 The number of classes in the transfer learning task
#             epochs : int
#                 How many epochs to update with
#             lr : float
#                 The learning rate
#             scheduler : bool
#                 If true, uses a learning rate scheduler while training
#             scheduler_step: int
#                 Scheduling step parameter
#             gamma: float
#                 Gamma value for scheduler
#             save: bool
#                 Whether or not to save results of update
#         """
#         target_model = torch.load(
#             f"{self.saved_models_dir}/shadow_model_{model_ind+1}",
#             map_location=self.device,
#         )
        
#         target_model._modules[last_layer_name] = torch.nn.Linear(in_features=target_model._modules[last_layer_name].in_features, out_features=num_classes)
        
#         for name, param in target_model.named_parameters():
#             found_layer_flag = False
    
#             for named_layer in named_layers:
#                 if named_layer in name:
#                     found_layer_flag = True
            
#             if not found_layer_flag:
#                 param.requires_grad = False
        
#         optimizer = torch.optim.SGD
#         model_wrap = ModelUtility(
#             model=target_model,
#             criterion=self.criterion,
#             optimizer=optimizer,
#             lr=lr,
#             scheduler=scheduler,
#             schedule_step=scheduler_step,
#             gamma=gamma,
#             out_features=self.output_dim,
#             device=self.device,
#             prefix=self.saved_models_dir,
#         )
        
#         dataloaders = {
#             "train": torch.utils.data.DataLoader(
#                 dataset=transfer_set, batch_size=self.batch_size, shuffle=True, num_workers=16, persistent_workers=True
#             )
#         }
        
#         out_model, _ = model_wrap.private_fit(
#             dataloaders=dataloaders,
#             num_epochs=epochs,
#             start_epoch=0,
#             epsilon=target_epsilon,
#             delta=target_delta,
#             C=5,
#             save=False,
#             train_only=True,
#             desc_string=f"Privately Transferring Model {model_ind+1}"
#         )
        
#         if manual_save_name:
#             manual_save_name = "_" + manual_save_name
#         else:
#             manual_save_name = ""
        
#         if save:
#             torch.save(out_model, f"{self.saved_models_dir}/shadow_model_{model_ind+1}_transfer_DP_eps={target_epsilon}" + manual_save_name)
#         return out_model

    
    def get_topk_logits(self, preds, k=3):
        classes = set(range(0, preds.shape[-1]))
        with torch.no_grad():
            mask_mult = torch.zeros_like(preds).bool()
            mask_add = torch.zeros_like(preds).bool()
            for idx in range(len(preds)):
                top_k = torch.topk(preds[idx], k)
                mask_mult[idx].scatter_(0, top_k.indices, 1)
                mask_add[idx].scatter_(0, torch.LongTensor(list(classes - set(top_k.indices.tolist()))), 1)

            leftover = ((1 - (mask_mult.float()*preds).sum(dim=1))/(len(classes)-k)).unsqueeze(1)*mask_add.float()
            new_preds = mask_mult*preds+leftover
        return new_preds
