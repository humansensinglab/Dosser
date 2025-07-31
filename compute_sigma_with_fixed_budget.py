import argparse
import torch
from privacy_analysis import *
from utils import get_dataset


def dosser_sigma(dataset, data_path, group_size=50, iterations=10000, epsilon=1.0):
    """
    Compute the noise magnitude (sigma) for a given privacy budget.
    
    Args:
        dataset (str): The name of the dataset.
        data_path (str): Path to the dataset.
        group_size (int): The group size for sampling. Default is 50.
        iterations (int): Number of sampling iterations. Default is 10000.
        epsilon (float): The target privacy budget (epsilon). Default is 0.8.
    
    Returns:
        float: The computed noise magnitude (sigma).
    """
    # Load dataset and its properties
    channel, im_size, num_classes, class_names, mean, std, \
        dst_train, dst_test, testloader = get_dataset(dataset, data_path)

    # Prepare class indices
    indices_class = [[] for _ in range(num_classes)]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    # Define alphas for RDP computation
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 2000))

    # Compute sampling probabilities
    qc = [group_size / len(indices_class[c]) for c in range(num_classes)]
    q = max(qc)

    # Binary search for sigma
    sigma_low, sigma_high = 0, 50
    rdps = [iterations * compute_omega(q, sigma_high, alpha) for alpha in alphas]
    eps_high, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=5e-6)
    epsilon_tolerance = 0.01

    while epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        rdps = [iterations * compute_omega(q, sigma, alpha) for alpha in alphas]
        eps, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=5e-6)

        if eps < epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute noise magnitude (sigma) for a given privacy budget.")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="The name of the dataset (e.g., CIFAR10).")
    parser.add_argument("--data_path", type=str, default="./data", help="The path to the dataset.")
    parser.add_argument("--group_size", type=int, default=50, help="The group size for sampling (default: 50).")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of sampling iterations (default: 10000).")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The target privacy budget epsilon (default: 1.0).")

    args = parser.parse_args()

    sigma = dosser_sigma(
        dataset=args.dataset,
        data_path=args.data_path,
        group_size=args.group_size,
        iterations=args.iterations,
        epsilon=args.epsilon
    )
    
    print(f"Computed noise magnitude (sigma): {sigma:.4f}")
