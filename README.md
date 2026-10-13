# Generative Causal Explainers (GCE) for Parkinson's Disease

Code for learning a latent representation of subthalamic local field potential
features in which one dimension is trained to detect deep brain stimulation
(stimulation-aligned, K = 1) and the remaining dimensions (auxiliary, L = 5) are
left free to carry other structure, including therapeutic state. The framework
follows Generative Causal Explanations (O'Shaughnessy et al., *NeurIPS* 2020):
a frozen ON/OFF classifier, a VAE, and a training objective that maximises the
information flow I(α; Ŷ) from the stimulation-aligned dimension to the
classifier output.

> **Code only.** Patient recordings, clinical scores, trained model weights and
> all result files are not included. See [Data availability](#data-availability).

## Installation

```bash
conda env create -f environment.yml
conda activate gce_vae
pip install -e .
```

Figure scripts use the Roboto font, and the feature-attribution panels use Myriad Pro;
matplotlib falls back to its default font if these are not installed.

## Repository layout

```
gce/             Core library: classifier, VAE, causal-effect estimator, data loading
training/        Classifier training and VAE hyperparameter search
analysis/        Latent-space analyses and supplementary statistics
paper_figures/   Scripts that render manuscript figure panels
config/          pd_all_save.yml (training configuration) and paths.py (data/model paths)
```

## Configuration

`config/pd_all_save.yml` holds the training configuration. Key parameters:

| Parameter | Meaning |
|-----------|---------|
| `n_alpha` (K) | Stimulation-aligned latent dimensions |
| `n_beta` (L) | Auxiliary latent dimensions |
| `lam_ml` (λ) | Weight of the VAE reconstruction term |
| `alpha_samples`, `beta_samples` | Monte Carlo samples for I(α; Ŷ) |
| `n_features` | Input features per sample (70) |

Training loss: `−I(α; Ŷ) + λ · (reconstruction + KL)`.

## Data availability

Electrophysiology recordings and clinical scores are not publicly available
owing to data-protection restrictions and the terms of patient consent.
De-identified derived data sufficient to reproduce the reported analyses are
available from the corresponding author on reasonable request, subject to a
data-sharing agreement and approval by the Medical Ethics Committee of the
Julius-Maximilians-University Würzburg.

## License

MIT; see [LICENSE](LICENSE).

## Acknowledgements

Builds on Generative Causal Explanations (O'Shaughnessy et al., *NeurIPS* 2020).
