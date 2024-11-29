import torch
import torch.nn as nn

class LogLikelihoodLossWithCensoring(nn.Module):
    def __init__(self, dist_type):
        """
        Initialize the log-likelihood loss with censoring class.
        
        Args:
            dist_type (str): Distribution type to use for survival prediction (currently only supports 'weibull').
        """
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.dist_type = dist_type

    def forward(self, inputs, survival_times, risk_status):
        """
        Compute the log-likelihood loss with censoring.
        
        Args:
            inputs (Tensor): Weibull parameters (alpha and lambda) predicted by the model, shape [batch_size, 2].
            survival_times (Tensor): Actual survival times for each sample, shape [batch_size].
            risk_status (Tensor): Binary censoring indicators (1 if event occurred, 0 if censored), shape [batch_size].

        Returns:
            loss (Tensor): Mean log-likelihood loss with censoring.
        """
        survival_times = torch.squeeze(survival_times)
        risk_status = torch.squeeze(risk_status)

        # Extract Weibull parameters from inputs
        alpha = inputs[:, 0]
        lam = inputs[:, 1]

        # Calculate the survival log-probabilities
        survival_term = -torch.pow(survival_times / lam, alpha)

        # Calculate the hazard function where risk_status is 1
        hazard_term = torch.log(alpha / lam) + (alpha - 1) * torch.log(survival_times / lam)

        # Debugging: Check for NaNs or Infs in intermediate calculations
        if torch.any(torch.isnan(survival_term)) or torch.any(torch.isinf(survival_term)):
            print("Warning: NaN or Inf detected in survival_term")
            print("alpha:", alpha)
            print("lam:", lam)
            print("survival_times:", survival_times)
            print("survival_term:", survival_term)

        if torch.any(torch.isnan(hazard_term)) or torch.any(torch.isinf(hazard_term)):
            print("Warning: NaN or Inf detected in hazard_term")
            print("alpha:", alpha)
            print("lam:", lam)
            print("survival_times:", survival_times)
            print("hazard_term:", hazard_term)

        # Apply censoring
        log_likelihood_loss = -torch.sum(hazard_term[risk_status > 0.5]) - torch.sum(survival_term)

        # Mean loss across batch
        loss = log_likelihood_loss / inputs.size(0)

        # Ensure finite loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered, check inputs for stability issues.")

        return loss


def contrastive_loss(latent_embeddings, labels, margin=1.0):
    """
    Contrastive loss for event vs. censored patients.
    latent_embeddings: Latent embeddings from the model's output.
    labels: True labels indicating if the sample is censored or event (1 for event, 0 for censored).
    """
    loss = 0
    num_pairs = 0  # To count the number of pairs

    for i in range(len(latent_embeddings)):
        for j in range(i + 1, len(latent_embeddings)):
            if labels[i] == labels[j]:
                # Pull together embeddings of the same class (event or censored)
                loss += torch.norm(latent_embeddings[i] - latent_embeddings[j], p=2)
            else:
                # Push apart embeddings of different classes (event vs censored)
                loss += torch.max(torch.tensor(0.0), margin - torch.norm(latent_embeddings[i] - latent_embeddings[j], p=2))
            
            num_pairs += 1  # Count the number of pairs considered

    # Divide the total loss by the number of pairs
    loss = loss / num_pairs if num_pairs > 0 else 0

    return loss