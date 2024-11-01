

import numpy as np

import matplotlib.pyplot as plt

from pycox.evaluation import EvalSurv
from statsmodels.stats.proportion import proportion_confint

# This warning type is removed due to pandas future warnings
# https://github.com/havakv/pycox/issues/162. Incompatibility between pycox and pandas' new version
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def bern_conf_interval(n, mean, ibs=False):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and not ibs:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)


def plot_model_losses(train_loss, val_loss, fig_path, title, x_label='Epochs'):
    plt.figure(figsize=(15, 15))
    plt.semilogy(train_loss, label='Train')
    plt.semilogy(val_loss, label='Valid')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.savefig(fig_path)
    # plt.show()
    plt.close()


def obtain_c_index(surv_f, time, censor):
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
    ci = ev.concordance_td()

    # Obtain also ibs
    time_grid = np.linspace(time.min(), time.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    return ci, ibs

def sample_from_dist(params, distributions):
    """Generates samples from specified distribution parameters."""
    samples = []
    for i, dist in enumerate(distributions):
        if dist == 'weibull':
            alpha, lam = params[:, 0], params[:, 1]
            weibull_samples = lam * (-torch.log(torch.rand_like(alpha)))**(1 / alpha)
            samples.append(weibull_samples.cpu().numpy())
        else:
            raise NotImplementedError(f"Unknown distribution type: {dist}")
    return np.concatenate(samples, axis=0)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def plot_latent_space(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode

    z_space = []
    risk_statuses = []

    with torch.no_grad():
        for images, survival_times, risk_status in data_loader:
            images = images.to(device)

            # Forward pass to get z (latent space)
            _, weibull_params, _ = model(images)
            z_space.append(weibull_params.detach().cpu().numpy())
            risk_statuses.append(risk_status.detach().cpu().numpy())

    z_space = np.concatenate(z_space, axis=0)
    risk_statuses = np.concatenate(risk_statuses, axis=0)

    # Dimensionality reduction
    if z_space.shape[1] > 2:
        pca = PCA(n_components=2)
        z_space_reduced = pca.fit_transform(z_space)
    else:
        z_space_reduced = z_space

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_space_reduced[:, 0], z_space_reduced[:, 1], c=risk_statuses, cmap='viridis', alpha=0.6)
    plt.title('Latent Space Visualization')
    plt.xlabel('Z Dimension 1')
    plt.ylabel('Z Dimension 2')
    plt.colorbar(scatter, label='Risk Status')
    plt.grid(True)
    plt.savefig('latent_space_visualization.png')
    plt.close()

# Call this function after training the model
plot_latent_space(model, train_loader, device)
