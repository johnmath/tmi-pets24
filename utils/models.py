import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

class MetaClassifierSet(torch.utils.data.Dataset):
    """Dataset for Transfer MI Metaclassifier"""

    def __init__(self, in_set, out_set):
        """
        in_set : torch.Tensor
            Collection of stacked tensors that represent the model confidences of the IN points
        out_set : torch.Tensor
            Collection of stacked tensors that represent the model confidences of the IN points
        """
        
        self.labels = torch.LongTensor([1]*len(in_set) + [0]*len(out_set))
        self.data =  torch.concat([in_set, out_set], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx], self.labels[idx]
    
    @staticmethod
    def evaluate_accuracy(model, data_loader, device):
        model.eval()
        total_correct = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
        return total_correct / len(data_loader.dataset)
    
    
    @staticmethod
    def train_metaclassifier(
        model,
        dataset,
        criterion=nn.CrossEntropyLoss(),
        gamma=0.9,
        scheduler_step=7,
        optimizer=torch.optim.Adam,
        lr=0.005,
        epochs=15,
        batch_size=64,
        device='cpu',
        verbose=False,
    ):
        """Helper function to train individual metaclassifiers
            Parameters
            ----------
                model : PyTorch Model
                    The metaclassifier model
                dataset : PyTorch Dataset
                    The metaclassifier's dataset. 
                    In TransferMI, the data are the output of the shadow models when queried on the target point. The labels are 0 (OUT) and 1 (IN)
                gamma : float
                    The learning rate scheduler parameter
                scheduler_step : int
                    Number of epochs before multiplying learning rate by gamma
                lr : float
                    The metaclassifier's learning rate
                epochs : int
                    The number of training iterations
                batch_size : int
                    The batch size for the metaclassifier's training loader
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = model.to(device)
        optimizer = optimizer(
            model.parameters(), lr=lr, weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            scheduler_step, 
            gamma
        )
        model.train()
        for i in range(1,epochs+1):
            # loss for debug
            running_loss = 0
            for (inputs, labels) in loader:
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            scheduler.step()
            
        if verbose:
            print(
                f"Metaclassifier Final Training Error: {running_loss/len(dataset):.4}\n"
                + f"Metaclassifier Final Training Accuracy: {MetaClassifierSet.evaluate_accuracy(model, loader, device)*100:.5}%"
            )
            
        model.eval()
        return model
    
    
class NeuralNet(nn.Module):
    """PyTorch implementation of a multilayer perceptron with ReLU activations"""
    
    def __init__(self, input_dim, layer_sizes=[64], num_classes=2, dropout=False):
        super(NeuralNet, self).__init__()
        self._input_dim = input_dim
        self._layer_sizes = layer_sizes
        layers = [nn.Linear(input_dim, layer_sizes[0])]
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout())
        
        # Initialize all layers according to sizes in list
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        
        # Wrap layers in ModuleList so PyTorch
        # can compute gradients
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class ModelUtility:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr=0.01,
        override_optimizer=None,
        scheduler=False,
        gamma=0.9,
        schedule_step=5,
        tmax=100,
        out_features=10,
        device="cpu",
        prefix="",
    ):
        """Utility for training deep learning models in PyTorch
        ...
        Attributes
        ----------
            model : PyTorch Model
                A PyTorch machine learning model
            dataloaders : dict
                Dictionary of DataLoaders with keys ["train", "test", "holdout"]
            criterion : torch.nn Loss Function
                The PyTorch loss function to train the model
            optimizer : torch.optim Optimizer
                The training procedure for the model's weights.
                **Give the __init__ as input. Do not call it**
            lr : float
                The learning rate of the optimizer
            scheduler (Optional) : bool
                Turn on StepLR scheduler
            gamma (Optional) : float
                Factor to decay learning rate by
            schedule_step (Optional) : int
                Number of steps to before multiplying LR by gamma
            tmax (Optional) : int
                If schedule_step and gamma are None, will use CosineAnnealingLR with Tmax=tmax
            out_features : int
                Number of class labels in data set
            device : str
                The device to train model on. Options are ["cpu", "cuda"]
            prefix (Optional) : str
                Prefix for save path
        """

        self.model = model
        self.lr = lr
        self.criterion = criterion
        if override_optimizer:
            self.optimizer = override_optimizer
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=self.lr,)
        
        self.device = device
        if scheduler:
            if gamma and schedule_step:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, schedule_step, gamma
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tmax)
        else:
            self.scheduler = scheduler
        self.prefix = prefix

    def standard_fit(
        self, 
        dataloaders, 
        num_epochs=100, 
        start_epoch=0, 
        save=True, 
        train_only=True, 
        desc_string=None
    ):
        """Trains the model for a desired number of epochs using the chosen optimizer,
        criterion, dataloaders, and scheduler.
        ...
        Parameters
        ----------
            dataloaders : dict{DataLoader}
                A dictionary of PyTorch dataloaders. For this training loop, the
                keys must be ["train", "test"]
            num_epochs : int
                The total number of training iterations (i.e. the number of
                full passes through the training and validation data)
            start_epoch : int
                Specify which epoch training starts from if training from
                a checkpoint model
            save : bool
                If true, saves a checkpoint of the model every epoch and train/test
                loss at the end of training
            train_only : bool
                If true, skips test set
        Returns
        ----------
            model : PyTorch Model
                The trained machine learning model
            epoch_loss : arr
                The list of training errors per epoch
            epoch_acc : arr
                The list of testing errors per epoch
        """

        self.model = self.model.to(self.device)
        
        epoch_loss = []
        epoch_acc = []
        
        if not desc_string:
            desc_string = "Training..."
        loop = tqdm(range(1, num_epochs + 1), desc=desc_string)
        for epoch in loop:
            
            phases = ["train"]
            if not train_only:
                phases.append("test")
            for phase in phases:

                # Allow gradients when in training phase
                if phase == "train":
                    self.model.train()
                    running_loss = 0.0
                    running_train_acc = 0.0

                # Freeze gradients when in testing phase
                elif phase == "test":
                    self.model.eval()
                    running_test_loss = 0.0
                    running_test_acc = 0.0
            
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    if phase == "train":
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            loss.backward()
                            self.optimizer.step()
                            running_loss += loss.item() * inputs.size(0)
                            running_train_acc += (outputs.argmax(dim=1) == labels).sum()

                    if phase == "test":
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            test_loss = self.criterion(outputs, labels)
                            running_test_loss += test_loss.item() * inputs.size(0)
                            running_test_acc += (outputs.argmax(dim=1) == labels).sum()

                if self.scheduler and phase == "train":
                    self.scheduler.step()
            epoch_loss.append(running_loss / len(dataloaders["train"].dataset))
            if not train_only:
                epoch_acc.append(running_loss / len(dataloaders["test"].dataset))
                
            dataset_length = len(dataloaders["train"].dataset)
            loop.set_postfix_str(f"Train Loss: {running_loss/dataset_length:.4f}, Train Acc: {100*running_train_acc/dataset_length:.4f}")
            
            # Zero-one Accuracy
            if not train_only:
                test_acc = self.evaluate_accuracy(dataloaders["test"])

            if save:
                self.save_model(self.model, epoch + start_epoch, dp=False)

        # Save stats as np.array's
        archictecture_name = str(type(self.model)).split(".")[-1].split("'")[0]
        if save:
            np.save(
                self.prefix
                + archictecture_name
                + f"_Checkpoints/Train_Loss_{num_epochs}-Epochs",
                epoch_loss,
            )
            if not train_only:
                np.save(
                    self.prefix
                    + archictecture_name
                    + f"_Checkpoints/Test_Loss_{num_epochs}-Epochs",
                    epoch_acc,
                )
        if not train_only:
            return self.model, epoch_loss, epoch_acc
        return self.model, epoch_loss

    def evaluate_accuracy(self, test_set):
        total_correct = 0
        for inputs, labels in test_set:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
        return total_correct / len(test_set.dataset)

    def save_model(self, model_to_save, epoch, dp=False):
        """Saves a snapshot of a model (as a .pth file) in training.
        This checkpoint can be loaded by calling torch.load(<PATH>)
        ...
        Parameters
        ----------
            model_to_save : torchvision model
                The desired machine learning model to save
            epoch : int
                The current epoch (used for filename)
            dp : bool
                Indicated whether the model was trained using
                DP-SGD or not to add a prefix to the file name
        """

        archictecture_name = str(type(model_to_save)).split(".")[-1].split("'")[0]
        if dp:
            archictecture_name = "DP_" + archictecture_name

        dir_to_save_at = self.prefix + archictecture_name + "_Checkpoints"
        if not os.path.exists(dir_to_save_at):
            os.mkdir(dir_to_save_at)

        file_to_save_at = (
            dir_to_save_at + "/" + archictecture_name + "_" + str(epoch) + ".pth"
        )
        torch.save(model_to_save, file_to_save_at)