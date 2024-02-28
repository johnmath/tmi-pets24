import torch
import numpy as np
import pandas as pd
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
# from opacus import PrivacyEngine
# from opacus.data_loader import DPDataLoader
# from opacus.validators import ModuleValidator
# from opacus.utils.batch_memory_manager import BatchMemoryManager

class TorchMLUtils:
    
    def __init__(self):
        pass
    
    @staticmethod
    def dataframe_to_torch_dataset(dataframe, using_ce_loss=False, class_label=None, verbose=False):
        """Convert a one-hot pandas dataframe to a PyTorch Dataset of Tensor objects"""
        
        new = dataframe.copy()
        if class_label:
            label = class_label
        else:
            label = list(new.columns)[-1]
            if verbose:
                print(f"Inferred that class label is '{label}' while creating dataloader")
        labels = torch.Tensor(pd.DataFrame(new[label]).values)
        del new[label]
        data = torch.Tensor(new.values)
        
        if using_ce_loss:
            # Fixes tensor dimension and float -> int if using cross entropy loss
            return torch.utils.data.TensorDataset(data, labels.squeeze().type(torch.LongTensor))
        else:
            return torch.utils.data.TensorDataset(data, labels)
    
    @staticmethod
    def dataset_to_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, persistent_workers=False):
        """Wrap PyTorch dataset in a Dataloader (to allow batch computations)"""
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             num_workers=num_workers, 
                                             shuffle=shuffle,
                                             persistent_workers=persistent_workers)
        return loader
    
    @staticmethod
    def dataframe_to_dataloader(dataframe, batch_size=256, num_workers=4, shuffle=True, persistent_workers=False, using_ce_loss=False, class_label=None):
        """Convert a pandas dataframe to a PyTorch Dataloader"""
        
        dataset = TorchMLUtils.dataframe_to_torch_dataset(dataframe, using_ce_loss=using_ce_loss, class_label=class_label)
        return TorchMLUtils.dataset_to_dataloader(dataset, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=shuffle,
                                                  persistent_workers=persistent_workers)
    
    @staticmethod
    def get_logits_torch(test_loader, model, device="cpu", middle_measure="mean", variance_adjustment=1):
        """Takes in a test dataloader + a trained model and returns the scaled logit values

            ...
            Parameters
            ----------
                test_loader : PyTorch Dataloader
                    The Pytorch Dataloader for Dtest
                model : torch.nn.Module (PyTorch Neural Network Model)
                    A trained model to be queried on the test data
                device : str
                    The processing device for training computations. 
                    Ex: cuda, cpu, cuda:0
                middle_measure : str
                    When removing outliters from the data, this is the 
                    "center" of the distribution that will be used.
                    Options are ["mean", "median"]
                variance_adjustment : float
                    The number of standard deviations away from the "center"
                    we want to keep.
            ...
            Returns
            -------
                logits_arr : np.array
                    An array containing the scaled model confidence values on the query set
        """
    
        n_samples = len(test_loader.dataset)
        logit_arr = np.zeros((n_samples,1))  

        max_conf = 1-1e-16
        min_conf = 0+1e-16

        model = model.to(device)

        y_prob = torch.Tensor([])
        y_test = torch.Tensor([])
        for d, l in test_loader:
            d = d.to(device)
            model.eval()
            with torch.no_grad():
                out = model(d)
                # Get class probabilities
                out = nn.functional.softmax(out, dim=1).cpu()
                y_prob = torch.concat([y_prob, out])
                y_test = torch.concat([y_test, l])

        y_prob, y_test = np.array(y_prob), np.array(y_test, dtype=np.uint8)

        if (np.sum(y_prob > max_conf)):
            indices = np.argwhere(y_prob > max_conf)
    #             print(indices)
            for idx in indices:
                r,c = idx[0],idx[1]
                y_prob[r][c] = y_prob[r][c] - 1e-16

        if (np.sum(y_prob < min_conf)):
            indices = np.argwhere(y_prob < min_conf)
            for idx in (indices):
                r,c = idx[0],idx[1]
                y_prob[r][c] = y_prob[r][c] + 1e-16

        possible_labels = len(y_prob[0])
        for sample_idx, sample in enumerate(zip(y_prob, y_test)):

            conf, label = sample
            selector = [True for _ in range(possible_labels)]
            selector[label] = False

            first_term = np.log(conf[label])
            second_term = np.log(np.sum(conf[selector]))

            logit_arr[sample_idx, 0] = (first_term - second_term)

        logit_arr = logit_arr.reshape(-1)

        if middle_measure == "mean":
            middle = logit_arr.mean()
        elif middle_measure == "median":
            middle = np.median(logit_arr)

        logit_arr = logit_arr[logit_arr > middle - variance_adjustment*logit_arr.std()] # Remove observations below the min_range
        logit_arr = logit_arr[logit_arr < middle + variance_adjustment*logit_arr.std()] # Remove observations above max range
        return  logit_arr
    
    
    @staticmethod
    def fit(dataloaders, 
            model, 
            epochs=100, 
            optim_init=optim.Adam, 
            optim_kwargs={"lr": 0.003, "weight_decay": 0.0001}, 
            criterion=nn.CrossEntropyLoss(), 
            device="cpu", 
            verbose=True, 
            train_only=True, 
            early_stopping=False, 
            tol=10e-6, 
            scheduler_init=None, 
            scheduler_kwargs={}):
        """Fits a PyTorch model to any given dataset

            ...
            Parameters
            ----------
                dataloaders : dict
                   Dictionary containing 2 PyTorch DataLoaders with keys "train" and 
                   "test" corresponding to the two dataloaders as values
                model : torch.nn.Module (PyTorch Neural Network Model)
                    The desired model to fit to the data
                epochs : int
                    Training epochs for shadow models
                optim_init : torch.optim init object
                    The init function (as an object) for a PyTorch optimizer. 
                    Note: Pass in the function without ()
                optim_kwargs : dict
                    Dictionary of keyword arguments for the optimizer
                    init function
                criterion : torch.nn Loss Function
                    Loss function used to train model
                device : str
                    The processing device for training computations. 
                    Ex: cuda, cpu, cuda:1
                verbose : bool
                    If True, prints running loss and accuracy values
                train_only : bool
                    If True, only uses "train" dataloader

            ...
            Returns
            -------
                model : torch.nn.Module (PyTorch Neural Network Model)
                    The trained model
                train_error : list
                    List of average training errors at each training epoch
                test_acc : list 
                    List of average test accuracy at each training epoch
        """

        model = model.to(device)
        optimizer = optim_init(model.parameters(), **optim_kwargs)
        
        if scheduler_init:
            scheduler = scheduler_init(optimizer, **scheduler_kwargs)
        
        train_error = []
        test_loss = []
        test_acc = []
        if train_only:
            phases = ["train"]
        else:
            phases = ["train", "test"]
        print("Training...")
        if verbose:
            print("-"*8)
        
        try:

            for epoch in range(1, epochs + 1):
                if verbose:
                    a = time.time()
                    print(f"Epoch {epoch}")

                running_train_loss = 0
                running_test_loss = 0 
                running_test_acc = 0
                
                for phase in phases:
                    if phase == "train":
                        model.train()
                        for (inputs, labels) in dataloaders[phase]:
                            optimizer.zero_grad()
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = model.forward(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            running_train_loss += loss.item() * inputs.size(0)

                    elif phase == "test":
                        model.eval()
                        with torch.no_grad():
                            for (inputs, labels) in dataloaders[phase]:
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                outputs = model.forward(inputs)
                                loss = criterion(outputs, labels)
                                running_test_loss += loss.item() * inputs.size(0)
                                running_test_acc += sum(torch.max(outputs, dim=1)[1] == labels).item()
                scheduler.step()
                                                
                train_error.append(running_train_loss/len(dataloaders["train"].dataset))
                
                if len(train_error) > 1 and early_stopping:
                    if abs(train_error[-1] - train_error[-2]) < tol:
                        print(f"Loss did not decrease by more than {tol}")
                        if not verbose:
                            print(f"Final Train Error: {train_error[-1]:.6}")
                        if not train_only:
                            return model, train_error, test_loss, test_acc 
                        else: 
                            return model, train_error
                
                if not train_only:
                    test_loss.append(running_test_loss/len(dataloaders["test"].dataset))
                    test_acc.append(running_test_acc/len(dataloaders["test"].dataset))
                if verbose:
                    b = time.time()
                    print(f"Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        print(f"Test Error: {test_loss[-1]:.6}")
                        print(f"Test Accuracy: {test_acc[-1]*100:.4}%")
                    print(f"Time Elapsed: {b - a:.4} seconds")
                    print("-"*8)
        except KeyboardInterrupt:
            if not verbose:
                print(f"Final Train Error: {train_error[-1]:.6}")
            if not train_only:
                return model, train_error, test_loss, test_acc
            else: 
                return model, train_error
        
        if not verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error
        
#     @staticmethod
#     def private_fit(
#         dataloaders,
#         model,
#         optim_init=optim.Adam,
#         optim_kwargs={"lr": 0.03},
#         epsilon=8,
#         delta=10e-5,
#         device="cpu",
#         train_only=True,
#         C=1.2,
#         epochs=50,
#         criterion=nn.CrossEntropyLoss(),
#         save=False,
#         save_prefix="",
#         verbose=False
#     ):

#         model.train()
#         optimizer = optim_init(model.parameters(), **optim_kwargs)

#         if not ModuleValidator.is_valid(model):
#             model = ModuleValidator.fix(model)
#             optimizer = optim_init(model.parameters(), **optim_kwargs)

#         model.to(device)

#         privacy_engine = PrivacyEngine()
#         model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
#             module=model,
#             optimizer=optimizer,
#             epochs=epochs,
#             data_loader=dataloaders["train"],
#             target_epsilon=epsilon,
#             target_delta=delta,
#             max_grad_norm=C
#         )
        
#         with BatchMemoryManager(data_loader=dataloader, 
#                                 max_physical_batch_size=512, 
#                                 optimizer=optimizer) as memory_safe_dl:
#             dataloaders["train"] = memory_safe_dl
            
#         epoch_loss = []
#         epoch_acc = []
        
#         if train_only: phases = ["train"]
#         else: phases = ["train", "test"]
        
#         for epoch in range(1, epochs + 1):
            
#             if verbose:
#                 print(f"Epoch {epoch}")
#                 print("-" * 10)
            
#             for phase in phases:

#                 # Allow gradients when in training phase
#                 if phase == "train":
#                     model.train()
#                     running_loss = 0.0

#                 # Freeze gradients when in testing phase
#                 elif phase == "test":
#                     model.eval()
#                     running_test_loss = 0.0
                
#                 for i, (inputs, labels) in enumerate(dataloaders[phase]):
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)

#                     optimizer.zero_grad()
#                     model.zero_grad()

#                     if phase == "train":
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)
#                         loss.backward()
#                         optimizer.step()
#                         running_loss += loss.item() * inputs.size(0)

#                     if phase == "test":
#                         with torch.no_grad():
#                             outputs = model(inputs)
#                             test_loss = criterion(outputs, labels)
#                             running_test_loss += test_loss.item() * inputs.size(0)

#             epoch_loss.append(running_loss / len(dataloaders["train"].dataset))
#             if not train_only:
#                 epoch_acc.append(running_loss / len(dataloaders["test"].dataset))

#             # Zero-one Accuracy
#             if not train_only:
#                 test_acc = TorchMLUtils.get_metrics(dataloaders["test"].dataset, dataset_type="torch")
            

#             epsilon = privacy_engine.get_epsilon(delta)
            
#             if verbose:
#                 if not train_only:
#                     print(
#                         f"Train Loss: {epoch_loss[epoch - 1]:.4}\n"
#                         f"Test Loss {epoch_acc[epoch-1]:.4}\n"
#                         f"Test Accuracy {test_acc*100:.4}%\n"
#                         f"(ε = {epsilon:.2f}, δ = {delta})\n"
#                     )
#                 else:
#                     print(
#                         f"Train Loss: {epoch_loss[epoch - 1]:.4}\n"
#                         f"(ε = {epsilon:.2f}, δ = {delta})\n"
#                 )
#         if train_only:
#             return model, epoch_loss
#         else:
#              return model, epoch_loss, epoch_acc
            
    @staticmethod
    def get_metrics_from_labels(yhat, y):
        tn, fp, fn, tp = confusion_matrix(np.array(y), np.array(yhat)).ravel()
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        precision = (tp)/(tp + fp)
        recall = (tp)/(tp + fn)
        print(f"Accuracy: {accuracy*100:.4}%")
        print(f"Precision: {precision*100:.4}%")
        print(f"Recall: {recall*100:.4}%")
        return accuracy, precision, recall
    
    @staticmethod
    def get_metrics(model, dataset, device="cpu", dataset_type="pandas", acc_only=False):
        """Returns (Accuracy, Precision, Recall)
        If acc_only==True, returns (y, y_hat)"""
    
        print("Evaluating metrics...")
        predictions = torch.Tensor([])
        all_labels = torch.Tensor([])
        if dataset_type == "pandas":
            dl = TorchMLUtils.dataframe_to_dataloader(dataset)
        elif dataset_type == "torch":
            dl = TorchMLUtils.dataset_to_dataloader(dataset, batch_size=256, num_workers=16)
        model = model.to(device)
        model.eval()
        for data, labels in dl:
            data = data.to(device)
            labels = labels.squeeze()
            outs = nn.Softmax(dim=1)(model(data))
            pred_labels = torch.argmax(outs, dim=1)
            predictions = torch.concat([pred_labels.cpu(), predictions])
            all_labels = torch.concat([labels.cpu(), all_labels])
        
        if acc_only:
            return all_labels, predictions
        
        tn, fp, fn, tp = confusion_matrix(np.array(all_labels), np.array(predictions)).ravel()
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        precision = (tp)/(tp + fp)
        recall = (tp)/(tp + fn)
        print(f"Accuracy: {accuracy*100:.4}%")
        print(f"Precision: {precision*100:.4}%")
        print(f"Recall: {recall*100:.4}%")
        return accuracy, precision, recall
    
    
    

        
        
        