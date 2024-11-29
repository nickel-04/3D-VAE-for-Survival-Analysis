import torch
import torch.nn as nn

class LogLikelihoodLossWithCensoring(nn.Module):
    def __init__(self, num_points=10):
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.num_points = num_points  # Fixed number of time points

    def forward(self, surv, survival_times, risk_status):
        """
        Compute the log-likelihood loss with censoring.

        Args:
            surv (Tensor): Predicted survival probabilities at all time points, shape [batch_size, num_points].
            survival_times (Tensor): Actual survival times for each sample, shape [batch_size].
            risk_status (Tensor): Binary censoring indicators (1 if event occurred, 0 if censored), shape [batch_size].

        Returns:
            loss (Tensor): Mean log-likelihood loss with censoring.
        """
        batch_size, num_times = surv.shape
        assert num_times == self.num_points, f"Expected {self.num_points} time points, but got {num_times}."

        # Define time points (equally spaced between min_time and max_time in training data)
        # This is assuming the times are evenly distributed between the min and max observed times
        min_time = torch.min(survival_times)
        max_time = torch.max(survival_times)
        times = torch.linspace(min_time, max_time, self.num_points)

        loss = 0.0

        for i in range(batch_size):
            observed_time = survival_times[i].item()

            # Find the closest time point to the observed survival time
            # The observed time will be matched to the closest time in the times array
            observed_index = torch.abs(times - observed_time).argmin()

            # Check if the event was observed or censored
            if risk_status[i] == 1:  # Event occurred
                loss -= torch.log(surv[i, observed_index])
            else:  # Censored
                loss -= torch.log(1 - surv[i, observed_index])

        return loss / batch_size





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