import torch
import time
import pandas as pd
import yaml
import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from gce.utils import *


def hybrid_search(classifier, train_data, test_data, config, writer,
                  K_range=1, L_range=2, lamb_range=0.05, lamb_step=0.005):
    """
    Hybrid approach: Run Algorithm 1 first, then grid search around the solution.

    Args:
        classifier: Pre-trained classifier
        train_data: Training dataloader
        test_data: Test dataloader
        config: Configuration dictionary
        writer: Tensorboard writer
        K_range: How far to search around K* (e.g., 1 means [K*-1, K*, K*+1])
        L_range: How far to search around L* (e.g., 2 means [L*-2, ..., L*+2])
        lamb_range: How far to search around λ* (e.g., 0.02 means [λ*-0.02, ..., λ*+0.02])
        lamb_step: Step size for lambda grid
    """
    print("HYBRID SEARCH: Phase 1 - Algorithm 1")

    # Phase 1: Run Algorithm 1 to find approximate optimal
    K_star, L_star, lamb_star, nll_best, causal_best = algorithm1(
        classifier, train_data, test_data, config, writer, save_models=False
    )

    print("HYBRID SEARCH: Phase 2 - Grid Search")
    print(f"Algorithm 1 found: K={K_star}, L={L_star}, λ={lamb_star:.3f}")
    print(f"Searching around this point...")

    # Phase 2: Define grid around the solution
    K_values = [k for k in range(max(1, K_star - K_range), K_star + K_range + 1)]
    L_values = [l for l in range(max(1, L_star - L_range), L_star + L_range + 1)]
    lamb_values = list(np.arange(
        max(0.001, lamb_star - lamb_range),
        lamb_star + lamb_range + lamb_step,
        lamb_step
    ))

    print(f"K values: {K_values}")
    print(f"L values: {L_values}")
    print(f"λ values: {[f'{l:.3f}' for l in lamb_values]}")

    # Run grid search
    results = grid_search(
        classifier, train_data, test_data, config, writer,
        K_values=K_values,
        L_values=L_values,
        lamb_values=lamb_values
    )

    return results


def grid_search(classifier, train_data, test_data, config, writer,
                K_values=[1, 2], L_values=[3, 4, 5, 6], lamb_values=None):
    """
    Grid search over K, L, and lambda values.
    """
    if lamb_values is None:
        lamb_values = list(np.arange(0.05, 0.15, 0.01))

    save_dir = os.path.join(config['save_dir'], config["folder"])
    os.makedirs(save_dir, exist_ok=True)
    model = config['model_name']

    results = []
    total_runs = len(K_values) * len(L_values) * len(lamb_values)
    current_run = 0
    start_time = time.time()

    for K in K_values:
        for L in L_values:
            for lamb in lamb_values:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Training K={K}, L={L}, lamb={lamb:.3f}")

                encoder, decoder, avg_nll, avg_causal, avg_total = \
                    train_and_test('both', K, L, lamb, classifier,
                                   train_data, test_data, config, writer)

                # Save model (same format as algorithm1)
                if config['save_vae']:
                    torch.save(encoder.state_dict(),
                               os.path.join(save_dir,
                                            f'{model}_encoder_best_K{K}_L{L}_lamb{lamb:.3f}_causal{avg_causal:.4f}.pt'))
                    torch.save(decoder.state_dict(),
                               os.path.join(save_dir,
                                            f'{model}_decoder_best_K{K}_L{L}_lamb{lamb:.3f}_causal{avg_causal:.4f}.pt'))

                results.append({
                    'K': K,
                    'L': L,
                    'lamb': lamb,
                    'nll': avg_nll.item() if torch.is_tensor(avg_nll) else avg_nll,
                    'causal': avg_causal.item() if torch.is_tensor(avg_causal) else avg_causal,
                    'total': avg_total.item() if torch.is_tensor(avg_total) else avg_total
                })

                elapsed = time.time() - start_time
                print(f"    NLL: {avg_nll:.4f}, Causal: {avg_causal:.4f}, Time: {elapsed:.1f}s")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'grid_search_results.csv'), index=False)

    # Print best models
    print("\n")
    print("Grid Search Complete")
    print(f"\nBest by Causal Score:")
    best_causal = results_df.loc[results_df['causal'].idxmin()]
    print(
        f"  K={int(best_causal['K'])}, L={int(best_causal['L'])}, λ={best_causal['lamb']:.3f}, causal={best_causal['causal']:.4f}")

    print(f"\nBest by NLL:")
    best_nll = results_df.loc[results_df['nll'].idxmin()]
    print(f"  K={int(best_nll['K'])}, L={int(best_nll['L'])}, λ={best_nll['lamb']:.3f}, nll={best_nll['nll']:.4f}")

    return results_df


def algorithm1(classifier, train_data, test_data, config, writer, save_models=True):
    """
    Original GCE Algorithm 1.

    Returns:
        K_best, L_best, lamb_best, nll_best, causal_best
    """
    save_dir = os.path.join(config['save_dir'], config["folder"])
    os.makedirs(save_dir, exist_ok=True)
    model = config['model_name']

    # step 1
    plateau = False
    K, L, lamb = 0, 0, 0.0
    nll_best = 10000.0
    start_time = time.time()
    optimize_on = 'nll'

    while (not plateau):
        L += 1
        _, _, average_nll, _, _ = \
            train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, config, writer)

        if average_nll < nll_best:
            nll_best = average_nll
        else:
            plateau = True

        print("step 1, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average nll: %f] time: %4.4f" %
              (K, L, lamb, average_nll, time.time() - start_time))

    print("step 1, finished, [best K: %d] [best L: %d] [best lamb: %f] [best loss: %f] time: %4.4f" %
          (K, L, lamb, nll_best, time.time() - start_time))

    # step 2 and 3
    plateau = False
    causal_best = 10000.0
    start_time = time.time()
    optimize_on = 'both'
    K_prev, L_prev, lamb_prev = 0, 0, 0.0

    while (not plateau):
        K_prev, L_prev, lamb_prev = K, L, lamb
        K += 1
        L -= 1
        lamb = 0.0
        approaches = False

        # step 3
        while (not approaches):
            lamb += 0.0025
            encoder_trained, decoder_trained, average_nll, average_causal, _ = \
                train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, config, writer)

            if average_nll < nll_best:
                approaches = True

            print(
                "step 2, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average nll: %f] time: %4.4f" %
                (K, L, lamb, average_nll, time.time() - start_time))

        if average_causal < causal_best:
            causal_best = average_causal
            # Track the best configuration
            K_best, L_best, lamb_best = K, L, lamb
        else:
            plateau = True

        # Save models if enabled
        if save_models and config['save_vae']:
            torch.save(encoder_trained.state_dict(),
                       os.path.join(save_dir,
                                    f'{model}_encoder_best_K{K}_L{L}_lamb{lamb:.3f}_causal{average_causal:.4f}.pt'))
            torch.save(decoder_trained.state_dict(),
                       os.path.join(save_dir,
                                    f'{model}_decoder_best_K{K}_L{L}_lamb{lamb:.3f}_causal{average_causal:.4f}.pt'))

        print("step 3, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average causal: %f] time: %4.4f" %
              (K, L, lamb, average_causal, time.time() - start_time))

    print("finished algorithm, [best K: %d] [best L: %d] [best lamb: %f] [best nll: %f] [best causal: %f] time: %4.4f" %
          (K_prev, L_prev, lamb_prev, nll_best, causal_best, time.time() - start_time))

    return K_prev, L_prev, lamb_prev, nll_best, causal_best


def train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, config, writer):
    z_dim = K + L
    config['z_dim'], config['n_alpha'], config['n_beta'] = z_dim, K, L
    params = {'K': K, 'L': L, 'lam': lamb}

    encoder, decoder = select_vae_model(config)
    encoder, decoder = encoder.to(config['device']), decoder.to(config['device'])
    optimizer = select_optimizer(config, encoder, decoder)

    encoder_trained, decoder_trained, average_nll, average_causal_effect, average_total = train(
        optimize_on, encoder, decoder, optimizer, classifier, train_data, test_data, lamb, config, params, writer
    )

    return encoder_trained, decoder_trained, average_nll, average_causal_effect, average_total


def train(optimize_on, encoder, decoder, optimizer, classifier, train_dataset, test_dataset, lamb, config, params,
          writer):
    for epoch in range(config['epochs']):
        total_loss = 0
        for imgs, targets in train_dataset:

            imgs, targets = imgs.to(config['device']), targets.to(config['device'])

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

            if optimize_on == 'nll':
                loss = nll

            elif optimize_on == 'causal':
                causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
                loss = causalEffect

            elif optimize_on == 'both':
                causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
                loss = config['use_causal'] * causalEffect + lamb * nll

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        log_metrics(writer, params, epoch, total_loss, mode="Train")
        average_nll, average_causal_effect, average_total = test(
            encoder, decoder, classifier, test_dataset, lamb, config, params, writer, epoch
        )

    return encoder, decoder, average_nll, average_causal_effect, average_total


def test(encoder, decoder, classifier, test_data, lamb, config, params, writer, epoch):
    encoder.eval()
    decoder.eval()
    total_nll = 0
    total_causal_effect = 0
    total_loss = 0
    batch = 0

    with torch.no_grad():
        for imgs, targets in test_data:
            imgs, targets = imgs.to(config['device']), targets.to(config['device'])

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)
            causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
            loss = config['use_causal'] * causalEffect + lamb * nll

            total_nll += nll
            total_causal_effect += causalEffect
            total_loss += loss

            batch += 1

        log_metrics(writer, params, epoch, total_loss, mode="Test")

        average_nll = total_nll / (batch + 1)
        average_causal_effect = total_causal_effect / (batch + 1)
        average_total = total_loss / (batch + 1)

    return average_nll, average_causal_effect, average_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pd_all_save.yml')
    parser.add_argument('--mode', default='hybrid',
                        choices=['algorithm1', 'grid', 'hybrid'],
                        help='Training mode: algorithm1, grid, or hybrid')

    # Grid search arguments
    parser.add_argument('--K', nargs='+', type=int, default=[1, 2],
                        help='K values for grid search')
    parser.add_argument('--L', nargs='+', type=int, default=[3, 4, 5, 6],
                        help='L values for grid search')
    parser.add_argument('--lamb_min', type=float, default=0.05,
                        help='Minimum lambda for grid search')
    parser.add_argument('--lamb_max', type=float, default=0.15,
                        help='Maximum lambda for grid search')
    parser.add_argument('--lamb_step', type=float, default=0.01,
                        help='Lambda step size for grid search')

    # Hybrid search arguments
    parser.add_argument('--K_range', type=int, default=1,
                        help='Search range around K* for hybrid mode')
    parser.add_argument('--L_range', type=int, default=2,
                        help='Search range around L* for hybrid mode')
    parser.add_argument('--lamb_range', type=float, default=0.02,
                        help='Search range around λ* for hybrid mode')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    config = to_vae_config(config)
    config = prepare_variables_pl(config)

    device = config['device']

    classifier = load_classifier_all(config)
    classifier.to(device)

    train_dataset, test_dataset = select_dataloader(config)

    log_writer = SummaryWriter()

    if args.mode == 'algorithm1':
        algorithm1(classifier, train_dataset, test_dataset, config, log_writer)

    elif args.mode == 'grid':
        lamb_values = list(np.arange(args.lamb_min, args.lamb_max, args.lamb_step))
        results = grid_search(
            classifier, train_dataset, test_dataset, config, log_writer,
            K_values=args.K,
            L_values=args.L,
            lamb_values=lamb_values
        )

    elif args.mode == 'hybrid':
        results = hybrid_search(
            classifier, train_dataset, test_dataset, config, log_writer,
            K_range=args.K_range,
            L_range=args.L_range,
            lamb_range=args.lamb_range,
            lamb_step=args.lamb_step
        )

    log_writer.flush()