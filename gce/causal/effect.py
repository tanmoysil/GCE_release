"""
Active causal effect implementation.

Computes mutual information I(α; Ŷ) between causal latent factors and
classifier output via Monte Carlo sampling.
"""
import numpy as np
import torch


def joint_uncond(params, decoder, classifier, device):
    """
    Sample-based estimate of joint, unconditional causal effect: -I(α; Ŷ).

    params keys used:
        alpha_samples, beta_samples, n_alpha, n_beta, z_dim,
        number_of_classes
    """
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['number_of_classes']).to(device)
    classifier.eval()
    for i in range(0, params['alpha_samples']):
        alpha = np.random.randn(params['n_alpha'])
        zs = np.zeros((params['beta_samples'], params['z_dim']))
        for j in range(0, params['beta_samples']):
            beta = np.random.randn(params['n_beta'])
            zs[j, :params['n_alpha']] = alpha
            zs[j, params['n_alpha']:] = beta

        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[1]
        p = 1. / float(params['beta_samples']) * torch.sum(yhat, 0)
        I = I + 1. / float(params['alpha_samples']) * torch.sum(torch.mul(p, torch.log(p + eps)))
        q = q + 1. / float(params['alpha_samples']) * p

    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    negCausalEffect = -I
    info = {"xhat": xhat, "yhat": yhat}
    return negCausalEffect, info


def joint_uncond_singledim(params, decoder, classifier, device, dim):
    """
    Sample-based estimate of causal effect for a single latent dim: -I(z_i; Ŷ).

    params keys used:
        alpha_samples, beta_samples, z_dim, number_of_classes
    dim: zero-indexed latent dimension
    """
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['number_of_classes']).to(device)
    classifier.eval()

    for i in range(0, params['alpha_samples']):
        z_fix = np.random.randn(1)
        zs = np.zeros((params['beta_samples'], params['z_dim']))
        for j in range(0, params['beta_samples']):
            zs[j, :] = np.random.randn(params['z_dim'])
            zs[j, dim] = z_fix

        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[1]
        p = 1. / float(params['beta_samples']) * torch.sum(yhat, 0)
        I = I + 1. / float(params['alpha_samples']) * torch.sum(torch.mul(p, torch.log(p + eps)))
        q = q + 1. / float(params['alpha_samples']) * p

    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    negCausalEffect = -I
    info = {"xhat": xhat, "yhat": yhat}
    return negCausalEffect, info


def informationFlow(config, decoder, classifier, device, Nalpha=None, Nbeta=None):
    """Compute I(z; Ŷ) — wrapper around joint_uncond."""
    if Nalpha is not None:
        config['alpha_samples'] = Nalpha
    if Nbeta is not None:
        config['beta_samples'] = Nbeta
    negI, _ = joint_uncond(config, decoder, classifier, device)
    return -1. * negI


def informationFlow_singledim(config, decoder, classifier, device, dims, Nalpha=None, Nbeta=None):
    """Compute {I(z_i; Ŷ) : i in dims} — wrapper around joint_uncond_singledim."""
    if Nalpha is not None:
        config['alpha_samples'] = Nalpha
    if Nbeta is not None:
        config['beta_samples'] = Nbeta
    ndims = len(dims)
    Is = np.zeros(ndims)
    for (i, dim) in enumerate(dims):
        negI, _ = joint_uncond_singledim(config, decoder, classifier, device, dim)
        Is[i] = -1. * negI.cpu().detach().numpy()
    return Is
