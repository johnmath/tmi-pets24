import numpy as np
from scipy.stats import norm, multivariate_normal
import torch
from .models import NeuralNet, MetaClassifierSet


class BaseAttack:
    
    def __init__(self, dataset, target_indices, device):
        self.dataset = dataset
        self.target_indices = target_indices
        self.device = device
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        pass
    
    @staticmethod
    def name():
        pass
    
    def run_attack(
        self, 
        target_model, 
        target_model_ind, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
    ):
        pass

class TMI(BaseAttack):
    
    def __init__(self, dataset, target_indices, device):
        super().__init__(dataset, target_indices, device)
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        return TMI(dataset, target_indices, device)
    
    @staticmethod
    def name():
        return "TMI"
    
    def setup_logits(self, challenge_pt, target_model_ind, target_point_logits, mask):
        # Target confidence
        observed_logits = target_point_logits[challenge_pt][target_model_ind]
        
        # Fix mask
        target_point_logits_without_target_model = np.delete(target_point_logits, target_model_ind, axis=1) 
        mask_without_target_model = np.delete(mask, target_model_ind, axis=1) 
        
        # Separate in/out points
        in_set = target_point_logits_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)]
        out_set = target_point_logits_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)]
        
        return observed_logits, in_set.flatten(0,1), out_set.flatten(0,1)
    
    def run_attack(
        self, 
        target_model, 
        target_model_ind, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
    ):
        
        observed_confidence, in_set, out_set = self.setup_logits(challenge_pt, target_model_ind, clipped_logits, mask)

        # Instantiate and Train Metaclassifier
        mclf_set = MetaClassifierSet(in_set, out_set)
        mclf = NeuralNet(mclf_set[0][0].shape[0], layer_sizes=[64], num_classes=2, dropout=True)
        out_model = MetaClassifierSet.train_metaclassifier(mclf, mclf_set, epochs=20, device=self.device)                
        
        out_model.eval()
        with torch.set_grad_enabled(False):
            mclf_pred = out_model(observed_confidence.to(self.device))        
            membership_score = torch.nn.Softmax()(mclf_pred).cpu()
        
        return torch.Tensor([membership_score[:, 1].mean()])

    
class OnlineLira(BaseAttack):
    
    def __init__(self, dataset, target_indices, device):
        super().__init__(dataset, target_indices, device)
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        return OnlineLira(dataset, target_indices, device)
    
    @staticmethod
    def name():
        return "White-Box Online Lira"
    
    def setup_logits(self, challenge_pt, target_model_ind, target_point_logits, mask):
        # Target confidence
        
        
        observed_logits = target_point_logits[challenge_pt][target_model_ind]

        label = self.dataset[self.target_indices[challenge_pt]][1]
        # print(observed_logits.shape, label)
        observed_confidence = observed_logits[label]
        
        
        # Fix mask
        target_point_logits_without_target_model = np.delete(target_point_logits, target_model_ind, axis=1) 
        mask_without_target_model = np.delete(mask, target_model_ind, axis=1) 
        
        # Separate in/out points
        in_set = target_point_logits_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)][:,label]
        out_set = target_point_logits_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)][:,label]
        
        return observed_confidence, in_set, out_set
        
    def run_attack(
        self, 
        target_model, 
        target_model_ind, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
    ):
        observed_confidence, in_set, out_set = self.setup_logits(challenge_pt, target_model_ind, target_point_scaled_logits, mask)
        
        # In distribution
        in_set = torch.Tensor(in_set)
        in_set = torch.nan_to_num(in_set, posinf=1e10, neginf=-1e10)
        in_set = in_set[torch.isfinite(in_set)]
        mean_in = torch.median(in_set).cpu()
        std_in = torch.std(in_set).cpu()

        # Out distribution
        out_set = torch.Tensor(out_set)
        out_set = torch.nan_to_num(out_set, posinf=1e10, neginf=-1e10)
        out_set = out_set[torch.isfinite(out_set)]
        mean_out = torch.median(out_set).cpu()
        std_out = torch.std(out_set).cpu()

        score_in = norm.logpdf(
            observed_confidence, loc=mean_in, scale=std_in + 1e-30
        )
        score_out = norm.logpdf(
            observed_confidence, loc=mean_out, scale=std_out + 1e-30
        )
        score = score_in - score_out
        
        return score
    
    
class AdaptedOnlineLira(BaseAttack):
    
    def __init__(self, dataset, target_indices, device):
        super().__init__(dataset, target_indices, device)
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        return AdaptedOnlineLira(dataset, target_indices, device)
    
    @staticmethod
    def name():
        return "Adapted LiRA"
    
    def setup_logits(self, challenge_pt, target_model_ind, target_point_logits, mask):
        # Target confidence
        observed_logits = target_point_logits[challenge_pt][target_model_ind]
        
        argmax_label = torch.argmax(observed_logits,dim=1)
        observed_confidence = observed_logits[torch.arange(observed_logits.shape[0]),argmax_label]
        
        # Fix mask
        target_point_logits_without_target_model = np.delete(target_point_logits, target_model_ind, axis=1) 
        mask_without_target_model = np.delete(mask, target_model_ind, axis=1) 
        
        # Separate in/out points
        
        in_set = target_point_logits_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)]
        out_set = target_point_logits_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)]
        
        in_set = in_set[:,torch.arange(in_set.shape[1]),argmax_label]
        out_set = out_set[:,torch.arange(out_set.shape[1]),argmax_label]
#         [:,argmax_label]
        return observed_confidence, in_set, out_set
        
    def run_attack(
        self, 
        target_model, 
        target_model_ind, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
    ):
        observed_confidence, in_set, out_set = self.setup_logits(challenge_pt, target_model_ind, target_point_scaled_logits, mask)
        # In distribution
        
        in_set = torch.Tensor(in_set)
        in_set = torch.nan_to_num(in_set, posinf=1e10, neginf=-1e10)
        in_set = in_set[torch.isfinite(in_set).all(dim=1)]
        mean_in = torch.mean(in_set,dim=0).cpu()
        
        cov_in = torch.cov(in_set.T).cpu()

        # Out distribution
        out_set = torch.Tensor(out_set)
        out_set = torch.nan_to_num(out_set, posinf=1e10, neginf=-1e10)
        out_set = out_set[torch.isfinite(out_set).all(dim=1)]
        mean_out = torch.mean(out_set,dim=0).cpu()
        cov_out = torch.cov(out_set.T).cpu()
        
        try:
            score_in = multivariate_normal.logpdf(
                observed_confidence, mean=mean_in, cov=cov_in, allow_singular=True,
            )

            score_out = multivariate_normal.logpdf(
                observed_confidence, mean=mean_out, cov=cov_out, allow_singular=True,
            )
            score = score_in - score_out
        except:
            score = 0
        return torch.Tensor([score])

class MultidimensionalOnlineLira(BaseAttack):
    
    def __init__(self, dataset, target_indices, device):
        super().__init__(dataset, target_indices, device)
        
    @staticmethod
    def create_attack(dataset, target_indices, device):
        return MultidimensionalOnlineLira(dataset, target_indices, device)
    
    @staticmethod
    def name():
        return "Multidimensional Online Lira"
    
    def setup_logits(self, challenge_pt, target_model_ind, target_point_logits, mask):
        # Target confidence
        observed_logits = target_point_logits[challenge_pt][target_model_ind]
        # argmax_label = torch.argmax(observed_logits)
        label = self.dataset[self.target_indices[challenge_pt]][1]
     
        observed_confidence = observed_logits[:,label]
        # print(observed_confidence.shape)
        # Fix mask
        target_point_logits_without_target_model = np.delete(target_point_logits, target_model_ind, axis=1) 
        
        
        mask_without_target_model = np.delete(mask, target_model_ind, axis=1) 
        
        # Separate in/out points
        in_set = target_point_logits_without_target_model[challenge_pt][mask_without_target_model[challenge_pt].astype(bool)][:,:,label]
        out_set = target_point_logits_without_target_model[challenge_pt][~mask_without_target_model[challenge_pt].astype(bool)][:,:,label]

        return observed_confidence.unsqueeze(0), in_set, out_set

    def run_attack(
        self, 
        target_model, 
        target_model_ind, 
        challenge_pt,
        target_point_logits, 
        target_point_scaled_logits,
        clipped_logits,
        raw_clipped_logits,
        mask,
    ):
        observed_confidence, in_set, out_set = self.setup_logits(challenge_pt, target_model_ind, clipped_logits, mask)

        # In distribution
        in_set = torch.Tensor(in_set)
        
        
        in_set = torch.nan_to_num(in_set, posinf=1e10, neginf=-1e10)
        in_set = in_set[torch.isfinite(in_set).all(dim=1)]
        mean_in = torch.mean(in_set, dim=0).cpu()
        cov_in = torch.cov(in_set.T).cpu()

        # Out distribution
        out_set = torch.Tensor(out_set)
        out_set = torch.nan_to_num(out_set, posinf=1e10, neginf=-1e10)
        out_set = out_set[torch.isfinite(out_set).all(dim=1)]
        mean_out = torch.mean(out_set, dim=0).cpu()
        cov_out = torch.cov(out_set.T).cpu()
        
        try:
            score_in = multivariate_normal.logpdf(
                observed_confidence, mean=mean_in, cov=cov_in, allow_singular=True,
            )

            score_out = multivariate_normal.logpdf(
                observed_confidence, mean=mean_out, cov=cov_out, allow_singular=True,
            )
        except:
            np.save("test_in_set.npy", np.array(in_set))
            np.save("test_out_set.npy", np.array(out_set))
            np.save("test_obs.npy", np.array(observed_confidence))
            np.save("test_logits.npy", np.array(target_point_scaled_logits))
            np.save("test_logits_raw.npy", np.array(raw_clipped_logits))
        score = score_in - score_out
        return torch.Tensor([score])


def AttackFactory(*arg):
    """
    Creates formatted attack dictionary ready for use
    
    *arg: Input as many BaseAttack as you need separated by commas
    """
    
    return {
        attack.name(): attack.create_attack 
        for attack in arg
    }
