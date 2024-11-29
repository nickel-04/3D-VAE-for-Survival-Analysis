import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from pycox.evaluation import EvalSurv
import torch

class ci_ibs:
    @staticmethod
    def bern_conf_interval(n, mean, ibs=False):
        """
        Calculate Bernoulli confidence interval using beta distribution.
        
        Args:
            n (int): Total number of observations.
            mean (float): Mean value.
            ibs (bool): Whether it is for integrated Brier score (IBS).

        Returns:
            tuple: (lower confidence bound, mean, upper confidence bound).
        """
        ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
        if mean < 0.5 and not ibs:
            ci_bot_2 = 1 - ci_top
            ci_top = 1 - ci_bot
            ci_bot = ci_bot_2
            mean = 1 - mean

        return np.round(ci_bot, 4), mean, np.round(ci_top, 4)

    @staticmethod
    def obtain_c_index(surv_f, time, censor):
        """
        Evaluate the survival predictions using the concordance index (C-index) and integrated Brier score (IBS).

        Args:
            surv_f (pd.DataFrame): Survival function predictions.
            time (np.ndarray): Array of survival times.
            censor (np.ndarray): Array of censoring indicators.

        Returns:
            tuple: C-index and integrated Brier score (IBS).
        """
        ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
        ci = ev.concordance_td('antolini')

        # Obtain also integrated Brier score
        time_grid = np.linspace(time.min(), time.max(), 100)
        ibs = ev.integrated_brier_score(time_grid)
        return ci, ibs

    def calculate_risk(self, time_train, surv, censor_val, num_points=10, device=None):
        """
        Calculate risk CDF at specified time points, compute C-index and IBS with confidence intervals.
        
        Parameters:
            time_train (torch.Tensor): Observed survival times in training data.
            surv (torch.Tensor): Predicted survival probabilities (0 to 1) for each time point and sample.
            censor_val (torch.Tensor): Censoring information (1 = event occurred, 0 = censored).
            num_points (int): Number of time points to evaluate the survival probabilities at.
            device (torch.device): Torch device (CPU or GPU).
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure inputs are in NumPy format
        time_train = time_train.detach().cpu().numpy()
        censor_val = censor_val.detach().cpu().numpy()
        surv = surv.detach().cpu().numpy()
        
        # Define a fixed set of time points to evaluate (you can customize this based on your data)
        # Example: Create `num_points` time intervals between min and max survival times
        min_time = np.min(time_train)
        max_time = np.max(time_train)
        times = np.linspace(min_time, max_time, num_points)

        # Calculate predicted risk for each sample at each time point
        pred_risk = 1 - surv  # Risk is 1 - survival probability

        # Calculate C-index and IBS along with confidence intervals
        ci, ibs = self.obtain_c_index(pd.DataFrame(pred_risk.T, index=times), 
                                        np.asarray(time_train), 
                                        np.asarray(censor_val))
        ci_conf_intervals = self.bern_conf_interval(len(np.asarray(time_train)), ci)
        ibs_conf_intervals = self.bern_conf_interval(len(np.asarray(time_train)), ibs, ibs=True)

        return ci_conf_intervals, ibs_conf_intervals