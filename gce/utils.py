from gce import dataset
import torch.nn as nn
import torch

from gce.models import classifier as models_classifiers, vae as models_vae
import numpy as np
from dcor import distance_correlation
import random


def select_dataloader(config):
    '''
    Selects a dataloader given the hyperparameters
    :param config: Set of hyperparameters
    :return: DataLoader
    '''

    if config["dataset"] == "pd_all":
        return dataset.get_pd_dataloaders(batch_size=config["batch_size"])

    elif config["dataset"] == "pd":
        return dataset.get_pd_groupkfold(batch_size=config["batch_size"], test_series=config['test_series'])

    elif config["dataset"] == "pd_all_save":
        return dataset.get_pd_dataloaders_with_save(batch_size=config["batch_size"])

    else:
        raise Exception("No valid dataset selected!")


def select_classifier(config):
    '''
    Selects a model given the hyperparameters
    :param config: Set of hyperparameters
    :return: nn.module
    '''

    if config["include_classes"] is None:
        output_dim = 10
    else:
        output_dim = len(config["include_classes"])
        # output_dim = 2

    if config['classifier'] == 'dummy':
        return models_classifiers.DummyClassifier()

    elif config['classifier'] == 'linClassifier':
        input_dim = config['n_features']
        #output_dim = 2
        return models_classifiers.linClassifier(input_dim, output_dim)

    else:
        raise Exception("No valid model selected!")


def select_optimizer(config, model, model_2=None):
    '''
    Selects an optimizer given the hyperparameters
    :param config: Set of hyperparameters
    :param model: model that we want to optimize
    :return: nn.optim.optimizer
    '''
    if model_2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model_2.parameters())

    if config["optimizer"] == "SGD":
        base_optimizer = torch.optim.SGD(params, lr=config["lr"],
                                         momentum=config["momentum"],
                                         weight_decay=config["weight_decay"],
                                         nesterov=True)
    elif config["optimizer"] == "Adam":
        base_optimizer = torch.optim.Adam(params, lr=config["lr"],
                                          betas=(config["b1"], config["b2"]),
                                          weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        base_optimizer = torch.optim.AdamW(params, lr=config["lr"],
                                           betas=(config["b1"], config["b2"]),
                                           weight_decay=config["weight_decay"])
    else:
        raise Exception("No valid optimizer selected!")

    # If L1 regularization is enabled, wrap with ElasticNetOptimizer
    if "l1_lambda" in config and config["l1_lambda"] > 0:
        return ElasticNetOptimizer(base_optimizer, config["l1_lambda"])
    else:
        return base_optimizer


class ElasticNetOptimizer:
    '''Wrapper around optimizer to implement L1 regularization via proximal gradient descent'''

    def __init__(self, optimizer, l1_lambda):
        self.optimizer = optimizer
        self.l1_lambda = l1_lambda
        # Copy optimizer attributes
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # First perform normal optimizer step
        self.optimizer.step()

        # Then apply proximal operator for L1 regularization
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                for param in group['params']:
                    if param.requires_grad:
                        # Proximal operator for L1 (soft thresholding)
                        threshold = self.l1_lambda * lr
                        param.data = torch.sign(param.data) * torch.clamp(
                            torch.abs(param.data) - threshold, min=0.0
                        )

    def __getattr__(self, name):
        # Delegate to base optimizer for any attributes not defined here
        return getattr(self.optimizer, name)

    def state_dict(self):
        # Include both optimizer state and wrapper attributes
        state_dict = self.optimizer.state_dict()
        state_dict['l1_lambda'] = self.l1_lambda
        return state_dict

    def load_state_dict(self, state_dict):
        # Load l1_lambda first then pass the rest to optimizer
        l1_lambda = state_dict.pop('l1_lambda', self.l1_lambda)
        self.l1_lambda = l1_lambda
        self.optimizer.load_state_dict(state_dict)


def select_vae_model(config):
    '''
    Selects vae encoder and decoder given the hyperparameters
    :param config: Set of hyperparameters
    :return: encoder, decoder
    '''
    if config["vae_model"] in ["mnist_cvae", "fmnist_cvae"]:
        encoder = models_vae.Encoder(config['z_dim'], 1, config["image_size"] ** 2)
        decoder = models_vae.Decoder(config['z_dim'], 1, config["image_size"] ** 2)
    elif config["vae_model"] == "cifar10_cvae":
        encoder = models_vae.Encoder_cifar(config['z_dim'], 3, config["image_size"] ** 2)
        decoder = models_vae.Decoder_cifar(config['z_dim'], 3, config["image_size"] ** 2)
    elif config["vae_model"] == "vanilla_Encoder":
        encoder = models_vae.vEncoder(config['n_features'], config['z_dim'])
        decoder = models_vae.vDecoder(config['n_features'], config['z_dim'])
    else:
        raise Exception("No valid encoder/decoder selected!")

    return encoder, decoder


def load_classifier(config):
    # Select the selected classifier architecture and load the pretrained parameters
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['model_name'] + (str(config['test_series']))))
    return classifier


def load_encoder_decoder(config):
    # Select the encoder, decoder and optimizer
    encoder, decoder = select_vae_model(config)

    encoder.load_state_dict(torch.load(config['encoder_path']))
    decoder.load_state_dict(torch.load(config['decoder_path']))
    return encoder, decoder


def save_model(model, path):
    torch.save(model.state_dict(), path)


def prepare_variables_pl(config):
    # The device to run the model on
    z_dim = config['n_alpha'] + config['n_beta']
    if config['vae_model'] != "vanilla_Encoder" :
        x_dim = config['image_size'] ** 2
    else:
        x_dim = config['n_features']


    if config['include_classes'] is None:
        n_classes = 10
    else:
        n_classes = len(config['include_classes'])

    params = {
        "number_of_classes": n_classes,
        "alpha_samples": config['alpha_samples'],
        "beta_samples": config['beta_samples'],
        "z_dim": z_dim,
        "n_alpha": config['n_alpha'],
        "n_beta": config['n_beta'],
        "channel_dimension": 1,
        "x_dim": x_dim
    }
    base_path = config['save_dir'] + config["model_name"]
    # config['encoder_path'] = base_path + '_encoder'
    # config['decoder_path'] = base_path + '_decoder'

    config = {**config, **params}
    return config


def VAE_LL_loss(Xbatch, Xest, logvar, mu):
    batch_size = Xbatch.shape[0]
    sse_loss = torch.nn.MSELoss(reduction='sum')  # sum of squared errors
    KLD = 1. / batch_size * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = 1. / batch_size * sse_loss(Xest, Xbatch)
    auto_loss = mse + KLD
    return auto_loss, mse, KLD


def reconstruction_loss(x_reconstructed, x):
    return nn.BCELoss(reduction='sum')(x_reconstructed, x) / x.size(0)


def kl_divergence_loss(mean, logvar):
    return ((mean ** 2 + logvar.exp() - 1 - logvar) / 2).mean()


def to_classifier_config(config):
    return {**config['classifier'], **config['general'], **config['dataset'], }


def to_vae_config(config):
    # return {**config['vae'], **config['general'], **config['dataset'], "classifier": config["classifier"]["classifier"]}
    return {**config['vae'], **config['general'], **config['dataset'], **config['classifier']}


def to_visualize_config(config):
    return {**config['vae'], **config['general'], **config['dataset'], "classifier": config["classifier"]["classifier"]}


def get_x_vals(val_loader, n_classes=2, n_for_each_class=4):
    ###

    # Here we get a sufficient number of samples to get the images from
    data, targets = next(iter(val_loader))
    for x, y in val_loader:
        data = torch.cat([data, x], dim=0)
        targets = torch.cat([targets, y], dim=0)

        # Here we make sure that we have all the classes and that we have enough samples
        if len(torch.unique(targets)) >= n_classes:
            y_val = targets.numpy()
            indices = []
            for i in range(n_classes):
                indices += list(np.where(y_val == i)[0])[:n_for_each_class]

            if indices == n_classes * n_for_each_class:
                break

    indices = torch.tensor(indices)
    x_val = torch.index_select(data, 0, indices)

    return x_val


def explain(self, x, zs_sweep, ):
    """
    Generate explanation for input x.
    :param x: input to explain, torch(nsamp,nrows,ncols,nchans)
    :param zs_sweep: array of latent space perturbations for explanation
    :return Xhats: output explanation, (nsamp,z_dim,len(zs_sweep),nrows,ncols,nchans)
    :return yhats: classifier output corresponding to each entry of Xhats
    """
    (nsamples, nrows, ncols, nchans) = x.shape
    Xhats = np.zeros((nsamples, self.K + self.L, len(zs_sweep), nrows, ncols, nchans))
    yhats = np.zeros((nsamples, self.K + self.L, len(zs_sweep)))
    for isamp in range(nsamples):
        x_torch = x[isamp:isamp + 1, :, :, :].permute(0, 3, 1, 2).float().to(self.device)
        z = self.encoder(x_torch)[0][0].detach().cpu().numpy()
        for latent_dim in range(self.K + self.L):
            for (iz, z_sweep) in enumerate(zs_sweep):
                ztilde = z.copy()
                ztilde[latent_dim] += z_sweep
                xhat = self.decoder(torch.unsqueeze(torch.from_numpy(ztilde), 0).to(self.device))
                yhat = np.argmax(self.classifier(xhat)[0].detach().cpu().numpy())
                img = xhat.permute(0, 2, 3, 1).detach().cpu().numpy()
                Xhats[isamp, latent_dim, iz, :, :, :] = img
                yhats[isamp, latent_dim, iz] = yhat
    return Xhats, yhats


def log_metrics(writer, params, steps, metrics, mode):

    # Extract hyperparameters
    K = params['K']
    L = params['L']
    lam = params['lam']

    # Create base tag without mode
    base_tag = f"Loss/K={K}_L={L}_lam={lam:.4f}"

    # Add mode as a separate series under the same base tag
    writer.add_scalars(
        base_tag,
        {mode: metrics},
        steps
    )


def load_classifier_all(config):
    # Select the selected classifier architecture and load the pretrained parameters
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['model_name']))
    return classifier


def dcor_test(x, y, num_permutations=1000):
    """
    Calculate distance correlation and p-value using permutation test

    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        num_permutations: Number of permutations for the test

    Returns:
        tuple: (correlation value, p-value)
    """

    # Original arrays must be the same length
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate actual distance correlation
    original_dcor = distance_correlation(x, y)

    # Permutation test
    permutation_dcors = []
    for _ in range(num_permutations):
        # Permute y values
        y_perm = np.random.permutation(y)
        # Calculate distance correlation with permuted data
        perm_dcor = distance_correlation(x, y_perm)
        permutation_dcors.append(perm_dcor)

    # Calculate p-value (proportion of permutation values >= original)
    p_value = (sum(perm_dcor >= original_dcor for perm_dcor in permutation_dcors)
               + 1) / (num_permutations + 1)

    return original_dcor, p_value