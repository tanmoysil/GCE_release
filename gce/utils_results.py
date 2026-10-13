

import torch
import os
import re
from dataclasses import dataclass
from typing import Dict, List
from gce.utils import *
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelParams:
    """
    Class to store VAE model parameters

    Attributes:
        subject_idx (int): Index of the subject (0-based)
        K (int): Number of latent dimensions
        L (int): Number of auxiliary dimensions
        lamb (float): Lambda parameter value
        causal_score (float): Causal score of the model
        filepath (str): Path to the model file
    """
    subject_idx: int
    K: int
    L: int
    lamb: float
    filepath: str


def parse_model_filename(filename: str) -> ModelParams:
    """
    Parse model filename to extract parameters

    Args:
        filename (str): Model filename in format
                       'model_XX_encoder_best_KY_LZ_lambW_causal-V'

    Returns:
        ModelParams: Object containing parsed parameters

    Example:
        >>> params = parse_model_filename('model_00_encoder_best_K1_L3_lamb0.010_causal-0.4860')
        >>> print(params.K)  # 1
        >>> print(params.L)  # 3
    """
    pattern = r'model_(\d+)_encoder_best_K(\d+)_L(\d+)_lamb([\d.]+)_causal-([\d.]+)'
    match = re.match(pattern, os.path.basename(filename))
    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")

    return ModelParams(
        subject_idx=int(match.group(1)),
        K=int(match.group(2)),
        L=int(match.group(3)),
        lamb=float(match.group(4)),
        filepath=filename
    )

def create_config_for_model(base_config: dict, model_params: ModelParams) -> dict:
    """
    Create a config for each model using base config and model-specific parameters

    Args:
        base_config (dict): Base configuration dictionary
        model_params (ModelParams): Model parameters object

    Returns:
        dict: Updated configuration dictionary
    """
    config = base_config.copy()

    # Update config with model-specific parameters
    config['n_alpha'] = model_params.K
    config['n_beta'] = model_params.L
    config['encoder_path'] = model_params.filepath
    config = prepare_variables_pl(config)
    return config


def load_encoder_for_params(base_config: dict, model_params: ModelParams):
    """
    Load a single encoder with the given parameters

    Args:
        base_config (dict): Base configuration dictionary
        model_params (ModelParams): Model parameters object

    Returns:
        torch.nn.Module: Loaded encoder model
    """
    # Create config for this specific model
    config = create_config_for_model(base_config, model_params)

    # Select and load encoder
    encoder, _ = select_vae_model(config)  # Ignore decoder return value
    encoder.load_state_dict(torch.load(model_params.filepath))
    encoder.eval()

    return encoder


def load_encoders(model_dir: str, base_config: dict) -> Dict[int, torch.nn.Module]:
    """
    Load encoders with correct mapping where model_XX is used for patient XX+1

    Args:
        model_dir (str): Directory containing encoder models
        base_config (dict): Base configuration dictionary

    Returns:
        Dict[int, torch.nn.Module]: Dictionary mapping patient numbers to their encoders

    Example:
        >>> encoders = load_encoders('./models', base_config)
        >>> patient_1_encoder = encoders[1]  # Gets encoder for patient 1
    """
    # Get all encoder files
    encoder_files = [f for f in os.listdir(model_dir)
                     if f.startswith("model_") and "encoder_best" in f]
    encoder_files = [os.path.join(model_dir, f) for f in encoder_files]

    # Parse parameters and sort by subject index
    params_list = [parse_model_filename(f) for f in encoder_files]
    params_list.sort(key=lambda x: x.subject_idx)

    # Load all encoders
    encoder_mapping = {}  # Maps patient number to encoder

    for params in params_list:
        encoder = load_encoder_for_params(base_config, params)
        # Map patient N to model_(N-1)
        patient_num = params.subject_idx + 1
        encoder_mapping[patient_num] = encoder

    return encoder_mapping

def extract_parameters(file_path):
    # Get just the filename without the path
    import os
    filename = os.path.basename(file_path)

    # Extract parameters using string operations
    params = {}

    # Extract K parameter
    if 'K' in filename:
        k_start = filename.find('K') + 1
        k_end = k_start
        while k_end < len(filename) and filename[k_end].isdigit():
            k_end += 1
        params['K'] = int(filename[k_start:k_end])

    # Extract L parameter
    if 'L' in filename:
        l_start = filename.find('L') + 1
        l_end = l_start
        while l_end < len(filename) and filename[l_end].isdigit():
            l_end += 1
        params['L'] = int(filename[l_start:l_end])

    # Extract lambda parameter
    if 'lamb' in filename:
        lamb_start = filename.find('lamb') + 4
        lamb_end = filename.find('_', lamb_start)
        if lamb_end == -1:
            lamb_end = len(filename)
        params['lambda'] = float(filename[lamb_start:lamb_end])

    return params


def get_latent_representations(encoder, data, patient_ids):
    """
    Get latent representations with patient-specific normalization

    Args:
        encoder: Trained VAE encoder
        data: Feature matrix (n_samples, n_features)
        patient_ids: Array of patient IDs for each sample (n_samples,)

    Returns:
        latent_dims: Latent representations (n_samples, K+L)
    """
    X_normalized = np.zeros_like(data, dtype=np.float32)

    for patient in np.unique(patient_ids):
        mask = patient_ids == patient
        if np.sum(mask) > 1:  # Need at least 2 samples for normalization
            scaler = StandardScaler()
            X_normalized[mask] = scaler.fit_transform(data[mask])
        else:
            # If only 1 sample, can't normalize (just keep as is)
            X_normalized[mask] = data[mask]

    X_tensor = torch.FloatTensor(X_normalized)
    with torch.no_grad():
        _, mu, _ = encoder(X_tensor)

    return mu.numpy()
