import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    get_time,
    DiffAugment,
    ParamDiffAug,
    poisson_sampling,
    number_sign_augment,
    Averager,
    convert_rgb_to_grayscale_and_resize
)
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from torchvision import transforms, datasets
from tqdm import trange
from pca import perform_pca_lowrank, project_new_data


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    dataset: str = 'CIFAR10'
    model: str = 'ConvNet'
    images_per_class: int = 50
    eval_mode: str = 'SS'
    num_experiments: int = 5
    num_evaluations: int = 1
    epoch_eval_train: int = 1000
    sampling_iterations: int = 10000
    training_iterations: int = 200000
    lr_images: float = 1.0
    lr_net: float = 0.01
    enable_pea: bool = True
    enable_ser: bool = True
    aux_dataset_path: str = '/data/runkai/PASDA/SD/sd_cifar10_50000_96/'
    aux_images_per_class: int = 100
    ser_dimension: int = 1000
    group_size: int = 50
    batch_train: int = 256
    dsa_strategy: str = 'color_crop_cutout_flip_scale_rotate'
    data_path: str = './data'
    save_path: str = 'result'
    max_norm: float = 1.0
    noise_sigma: float = 4.2969 # CIFAR-10 (1, 1e-5)
    device: str = None
    method: str = 'DM'
    dsa: bool = True
    dsa_param: object = None
    
    def __post_init__(self):
        """Initialize derived attributes after object creation."""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.dsa_param is None:
            from utils import ParamDiffAug
            self.dsa_param = ParamDiffAug()
        self.dsa = self.dsa_strategy.lower() != 'none'


class DatasetManager:
    """Manages dataset loading and organization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self._setup_directories()
        self._load_dataset()
        self._organize_real_data()
        
    def _setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.data_path, exist_ok=True)
        os.makedirs(self.config.save_path, exist_ok=True)
    
    def _load_dataset(self):
        """Load and prepare the dataset."""
        (self.channels, self.image_size, self.num_classes, self.class_names, 
         self.mean, self.std, self.training_dataset, self.testing_dataset, 
         self.test_loader) = get_dataset(self.config.dataset, self.config.data_path)
        
        self.eval_model_pool = get_eval_pool(self.config.eval_mode, 
                                           self.config.model, self.config.model)
        
        # Setup evaluation iteration pool
        if self.config.eval_mode in ['S', 'SS', 'B']:
            self.eval_iteration_pool = np.arange(1000, self.config.training_iterations + 1, 1000).tolist()
        else:
            self.eval_iteration_pool = [self.config.training_iterations]
    
    def _organize_real_data(self):
        """Organize real dataset into class-based structure."""
        # Load all images and labels
        all_images = [torch.unsqueeze(self.training_dataset[i][0], dim=0) 
                     for i in range(len(self.training_dataset))]
        all_labels = [self.training_dataset[i][1] for i in range(len(self.training_dataset))]
        
        # Organize by class
        self.class_indices = [[] for _ in range(self.num_classes)]
        for index, label in enumerate(all_labels):
            self.class_indices[label].append(index)
        
        self.real_images = torch.cat(all_images, dim=0).to(self.device)
        self.real_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        
        # Print dataset statistics
        for class_idx in range(self.num_classes):
            print(f'Class {class_idx}: {len(self.class_indices[class_idx])} real images')
        
        # Print channel statistics
        for ch in range(self.channels):
            channel_mean = torch.mean(self.real_images[:, ch]).item()
            channel_std = torch.std(self.real_images[:, ch]).item()
            print(f'Real images channel {ch}, mean = {channel_mean:.4f}, std = {channel_std:.4f}')
    
    def sample_images_by_class(self, class_label: int, num_samples: int) -> torch.Tensor:
        """Sample random images from the specified class using Poisson sampling."""
        total_samples = len(self.class_indices[class_label])
        sampling_rate = float(num_samples) / float(total_samples)
        sampled_indices = poisson_sampling(total_samples, sampling_rate)
        shuffled_indices = np.random.permutation(self.class_indices[class_label])[sampled_indices]
        return self.real_images[shuffled_indices]


class AuxiliaryDataManager:
    """Manages auxiliary dataset for SER (Subspace discovery for Error Reduction)."""
    
    def __init__(self, config: TrainingConfig, num_classes: int, device: str):
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.auxiliary_images = None
        
        if config.enable_ser:
            self._load_auxiliary_data()
    
    def _load_auxiliary_data(self):
        """Load auxiliary dataset based on the main dataset type."""
        transform = self._get_transform_for_dataset()
        auxiliary_dataset = datasets.ImageFolder(self.config.aux_dataset_path, transform=transform)
        
        # Organize auxiliary images by class
        auxiliary_images_by_class = [[] for _ in range(self.num_classes)]
        for image, label in auxiliary_dataset:
            if len(auxiliary_images_by_class[label]) < self.config.aux_images_per_class:
                auxiliary_images_by_class[label].append(image)
        
        # Stack and move to device
        self.auxiliary_images = torch.cat([torch.stack(aux_images) for aux_images in auxiliary_images_by_class]).to(self.device)
        
        # Convert to grayscale for MNIST datasets
        if self.config.dataset in ["MNIST", "FashionMNIST"]:
            self.auxiliary_images = convert_rgb_to_grayscale_and_resize(self.auxiliary_images)
    
    def _get_transform_for_dataset(self) -> transforms.Compose:
        """Get appropriate transform based on dataset."""
        if self.config.dataset == "CIFAR10":
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.config.dataset in ["TinyImageNet", "ImageNette"]:
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.config.dataset in ["MNIST", "FashionMNIST"]:
            return transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")


class TrainingSignalPrecomputer:
    """Precomputes training signals for efficiency."""
    
    def __init__(self, config: TrainingConfig, dataset_manager: DatasetManager, 
                 aux_data_manager: AuxiliaryDataManager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.aux_data_manager = aux_data_manager
        self.device = dataset_manager.device
        
        # Use DSA parameters from config
        self.dsa_param = config.dsa_param
        self.enable_dsa = config.dsa
        
        # Precomputed data
        self.networks = []
        self.real_output_sums = []
        self.seeds = []
        self.pca_components = []
        
    def precompute_training_signals(self):
        """Precompute all training signals for efficiency."""
        print("Precomputing training signals...")
        start_time = time.time()
        
        for _ in trange(self.config.sampling_iterations, desc='Precomputing signals'):
            # Create and prepare network
            network = self._create_network()
            self.networks.append(network)
            
            # Get embedding layer
            embedding_layer = self._get_embedding_layer(network)
            
            # Compute PCA components if SER is enabled
            pca_components = None
            if self.config.enable_ser:
                pca_components = self._compute_pca_components(embedding_layer)
                self.pca_components.append(pca_components)
            
            # Compute real output sums for each class
            real_output_sums_class = []
            seeds_class = []
            
            for class_label in range(self.dataset_manager.num_classes):
                # Sample real images
                real_images = self.dataset_manager.sample_images_by_class(
                    class_label, self.config.group_size)
                
                # Apply DSA if enabled
                seed = None
                if self.enable_dsa and self.config.images_per_class != 1:
                    seed = int(time.time() * 1000) % 100000
                    real_images = DiffAugment(real_images, self.config.dsa_strategy, 
                                            seed=seed, param=self.dsa_param)
                    seeds_class.append(seed)
                
                # Compute embeddings
                real_embeddings = embedding_layer(real_images).detach().cpu()
                
                # Apply SER projection if enabled
                if self.config.enable_ser and pca_components is not None:
                    real_embeddings = project_new_data(real_embeddings, 
                                                     pca_components[0], pca_components[1])
                
                # Compute normalized output sum
                real_output_sum = self._compute_normalized_output_sum(real_embeddings)
                real_output_sums_class.append(real_output_sum)
            
            self.real_output_sums.append(real_output_sums_class)
            self.seeds.append(seeds_class)
            
            # Move network to CPU to save GPU memory
            network.cpu()
        
        print(f'Precomputation completed in {time.time() - start_time:.2f} seconds')
    
    def _create_network(self) -> nn.Module:
        """Create and prepare a network."""
        network = get_network(self.config.model, self.dataset_manager.channels, 
                            self.dataset_manager.num_classes, self.dataset_manager.image_size)
        network = network.to(self.device)
        network.train()
        
        # Freeze parameters
        for param in network.parameters():
            param.requires_grad = False
        
        return network
    
    def _get_embedding_layer(self, network: nn.Module) -> nn.Module:
        """Get the embedding layer from the network."""
        if torch.cuda.device_count() > 1:
            return network.module.embed
        return network.embed
    
    def _compute_pca_components(self, embedding_layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PCA components for SER."""
        aux_embeddings = embedding_layer(self.aux_data_manager.auxiliary_images)
        principal_components, embedding_mean = perform_pca_lowrank(aux_embeddings, self.config.ser_dimension)
        return principal_components.cpu(), embedding_mean.cpu()
    
    def _compute_normalized_output_sum(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute normalized output sum for embeddings."""
        l2_norms = torch.norm(embeddings, dim=1).detach()
        scalers = torch.min(self.config.max_norm * torch.ones_like(l2_norms) / l2_norms, 
                           torch.ones_like(l2_norms))
        return torch.sum(scalers.unsqueeze(1) * embeddings, dim=0)


class SyntheticDataGenerator:
    """Generates and optimizes synthetic data."""
    
    def __init__(self, config: TrainingConfig, dataset_manager: DatasetManager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.device = dataset_manager.device
        
        # Initialize synthetic data
        self.synthetic_images = None
        self.synthetic_labels = None
        self.parameter_averager = None
        self.optimizer = None
        
        self._initialize_synthetic_data()
    
    def _initialize_synthetic_data(self):
        """Initialize synthetic images and labels."""
        # Initialize synthetic images
        self.synthetic_images = torch.randn(
            size=(self.dataset_manager.num_classes * self.config.images_per_class, 
                  self.dataset_manager.channels, 
                  self.dataset_manager.image_size[0], 
                  self.dataset_manager.image_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=self.device
        )
        
        # Initialize synthetic labels
        self.synthetic_labels = torch.tensor(
            np.array([np.ones(self.config.images_per_class) * i 
                     for i in range(self.dataset_manager.num_classes)]),
            dtype=torch.long,
            requires_grad=False,
            device=self.device
        ).view(-1)
        
        # Initialize parameter averager and optimizer
        self.parameter_averager = Averager([self.synthetic_images], alpha=0.99)
        self.optimizer = torch.optim.SGD([self.synthetic_images], 
                                        lr=self.config.lr_images, momentum=0.5)
        self.optimizer.zero_grad()
        
        print('Initialized synthetic data from random noise')
    
    def train_step(self, signal_precomputer: TrainingSignalPrecomputer):
        """Perform one training step."""
        if 'BN' in self.config.model:
            raise SystemError("Batch Normalization cannot be used in Differentially Private Dataset Condensation (DPDC)")
        
        # Sample a random precomputed signal
        sample_index = random.choice(range(self.config.sampling_iterations))
        selected_network = signal_precomputer.networks[sample_index].to(self.device)
        embedding_layer = signal_precomputer._get_embedding_layer(selected_network)
        
        total_loss = 0.0
        
        # Update synthetic data for each class
        for class_label in range(self.dataset_manager.num_classes):
            # Get synthetic images for this class
            start_idx = class_label * self.config.images_per_class
            end_idx = (class_label + 1) * self.config.images_per_class
            synthetic_batch = self.synthetic_images[start_idx:end_idx].reshape(
                (self.config.images_per_class, self.dataset_manager.channels, 
                 self.dataset_manager.image_size[0], self.dataset_manager.image_size[1]))
            
            # Apply PEA if enabled
            if self.config.enable_pea:
                synthetic_labels_batch = self.synthetic_labels[start_idx:end_idx].reshape(self.config.images_per_class)
                synthetic_batch, synthetic_labels_batch = number_sign_augment(synthetic_batch, synthetic_labels_batch)
            
            # Apply DSA if enabled
            if signal_precomputer.enable_dsa and self.config.images_per_class != 1:
                seed = signal_precomputer.seeds[sample_index][class_label]
                synthetic_batch = DiffAugment(synthetic_batch, self.config.dsa_strategy, 
                                            seed=seed, param=signal_precomputer.dsa_param)
            
            # Get real output sum and add noise
            real_output_sum = signal_precomputer.real_output_sums[sample_index][class_label].to(self.device)
            real_output_sum_noisy = (real_output_sum + 
                                   self.config.noise_sigma * torch.randn_like(real_output_sum) * self.config.max_norm)
            
            # Compute synthetic output
            synthetic_output = embedding_layer(synthetic_batch)
            
            # Apply SER projection if enabled
            if self.config.enable_ser:
                pca_components = signal_precomputer.pca_components[sample_index]
                synthetic_output = project_new_data(synthetic_output, 
                                                 pca_components[0].to(self.device),
                                                 pca_components[1].to(self.device))
            
            # Normalize synthetic output
            l2_norms = torch.norm(synthetic_output, dim=1).detach()
            scalers = torch.min(self.config.max_norm * torch.ones_like(l2_norms) / l2_norms, 
                               torch.ones_like(l2_norms))
            synthetic_output_scaled = scalers.unsqueeze(1) * synthetic_output
            
            # Apply balance factor
            balance_factor = float(self.config.group_size) / len(synthetic_output)
            synthetic_output_scaled = balance_factor * torch.sum(synthetic_output_scaled, dim=0)
            
            # Compute loss
            loss = torch.sum((real_output_sum_noisy - synthetic_output_scaled) ** 2) / (self.config.max_norm ** 2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update parameter averager
        self.parameter_averager.update([self.synthetic_images])
        
        return total_loss / self.dataset_manager.num_classes
    
    def get_evaluation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get synthetic data for evaluation."""
        synthetic_images_eval = copy.deepcopy(self.synthetic_images.detach())
        synthetic_labels_eval = copy.deepcopy(self.synthetic_labels.detach())
        
        # Apply parameter averaging
        averaged_parameters = self.parameter_averager.get_average_params()
        with torch.no_grad():
            synthetic_images_eval.copy_(averaged_parameters[0])
        
        return synthetic_images_eval, synthetic_labels_eval


class Evaluator:
    """Handles model evaluation."""
    
    def __init__(self, config: TrainingConfig, dataset_manager: DatasetManager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.device = dataset_manager.device
        self.accuracies_all_experiments = {model_key: [] for model_key in dataset_manager.eval_model_pool}
    
    def evaluate_synthetic_data(self, synthetic_images: torch.Tensor, 
                              synthetic_labels: torch.Tensor, 
                              experiment: int, iteration: int) -> None:
        """Evaluate synthetic data on all evaluation models."""
        for eval_model_key in self.dataset_manager.eval_model_pool:
            print(f'-------------------------\nEvaluation\nModel Train = {self.config.model}, '
                  f'Model Eval = {eval_model_key}, Iteration = {iteration}')
            
            print(f'DSA Augmentation Strategy: {self.config.dsa_strategy}')
            
            evaluation_accuracies = []
            
            for eval_run in range(self.config.num_evaluations):
                # Create evaluation network
                eval_network = get_network(eval_model_key, self.dataset_manager.channels, 
                                         self.dataset_manager.num_classes, 
                                         self.dataset_manager.image_size).to(self.device)
                
                # Prepare evaluation data
                synthetic_images_eval = synthetic_images.cpu()
                synthetic_labels_eval = synthetic_labels.cpu()
                
                # Apply PEA if enabled
                if self.config.enable_pea:
                    synthetic_images_eval, synthetic_labels_eval = number_sign_augment(
                        synthetic_images_eval, synthetic_labels_eval)
                
                # Evaluate
                _, train_accuracy, test_accuracy = evaluate_synset(
                    eval_run, eval_network, synthetic_images_eval, synthetic_labels_eval, 
                    self.dataset_manager.test_loader, self.config)
                
                evaluation_accuracies.append(test_accuracy)
            
            # Print results
            mean_acc = np.mean(evaluation_accuracies)
            std_acc = np.std(evaluation_accuracies)
            print(f'Evaluate {len(evaluation_accuracies)} random {eval_model_key} models, '
                  f'Mean = {mean_acc:.4f}, Std = {std_acc:.4f}\n-------------------------')
            
            # Store results for final iteration
            if iteration == self.config.training_iterations:
                self.accuracies_all_experiments[eval_model_key].extend(evaluation_accuracies)
    
    def save_visualization(self, synthetic_images: torch.Tensor, 
                          experiment: int, iteration: int, result_dir: str = None) -> None:
        """Save visualization of synthetic images in structured directory."""
        # Use provided result_dir or create a default one
        if result_dir is None:
            result_dir = self.config.save_path
        
        # Create visualization subdirectory
        vis_dir = os.path.join(result_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create structured filename
        visualization_filename = os.path.join(
            vis_dir,
            f'exp_{experiment:02d}_iter_{iteration:06d}_synthetic_images.png'
        )
        
        # Prepare images for visualization
        synthetic_images_visual = copy.deepcopy(synthetic_images.detach().cpu())
        
        # Denormalize
        for ch in range(self.dataset_manager.channels):
            synthetic_images_visual[:, ch] = (synthetic_images_visual[:, ch] * 
                                            self.dataset_manager.std[ch] + 
                                            self.dataset_manager.mean[ch])
        
        # Clip to valid range
        synthetic_images_visual = torch.clamp(synthetic_images_visual, 0.0, 1.0)
        
        # Select images for visualization
        vis_per_class = min(10, self.config.images_per_class)
        vis_indices = np.array([
            np.arange(j * self.config.images_per_class, vis_per_class + j * self.config.images_per_class) 
            for j in range(self.dataset_manager.num_classes)
        ]).flatten()
        
        # Save image
        save_image(synthetic_images_visual[vis_indices, :], visualization_filename, 
                  nrow=vis_per_class)
        
        print(f'Visualization saved: {visualization_filename}')
    
    def print_final_results(self) -> None:
        """Print final results across all experiments."""
        print('\n==================== Final Results ====================\n')
        for model_key in self.dataset_manager.eval_model_pool:
            experiment_accuracies = self.accuracies_all_experiments[model_key]
            mean_acc = np.mean(experiment_accuracies) * 100
            std_acc = np.std(experiment_accuracies) * 100
            print(f'Ran {self.config.num_experiments} experiments, trained on {self.config.model}, '
                  f'evaluated with {len(experiment_accuracies)} random {model_key}, '
                  f'Mean Accuracy = {mean_acc:.2f}%, Standard Deviation = {std_acc:.2f}%')


class DosserTrainer:
    """Main trainer class for the Dosser algorithm."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.aux_data_manager = AuxiliaryDataManager(config, self.dataset_manager.num_classes, 
                                                   self.dataset_manager.device)
        self.signal_precomputer = TrainingSignalPrecomputer(config, self.dataset_manager, 
                                                          self.aux_data_manager)
        self.evaluator = Evaluator(config, self.dataset_manager)
        
        self.saved_data = []
        self.training_history = {
            'losses': [],
            'iterations': [],
            'evaluation_accuracies': {},
            'timestamps': []
        }
    
    def run_experiments(self):
        """Run all experiments."""
        # Precompute training signals
        self.signal_precomputer.precompute_training_signals()
        
        # Run experiments
        for experiment in range(self.config.num_experiments):
            print(f'\n================== Experiment {experiment} ==================\n')
            print('Hyper-parameters:', self.config.__dict__)
            print('Evaluation model pool:', self.dataset_manager.eval_model_pool)
            
            # Initialize synthetic data generator
            synthetic_generator = SyntheticDataGenerator(self.config, self.dataset_manager)
            print(f'{get_time()} Training begins')
            
            # Training loop
            for iteration in trange(self.config.training_iterations + 1, desc='Training Iterations'):
                # Evaluate if needed
                if iteration in self.dataset_manager.eval_iteration_pool:
                    synthetic_images_eval, synthetic_labels_eval = synthetic_generator.get_evaluation_data()
                    self.evaluator.evaluate_synthetic_data(synthetic_images_eval, synthetic_labels_eval, 
                                                         experiment, iteration)
                    
                    # Create result directory for this experiment if it doesn't exist
                    if not hasattr(self, '_current_result_dir'):
                        self._current_result_dir = self._create_result_directory()
                    
                    # Save visualization in structured directory
                    self.evaluator.save_visualization(synthetic_images_eval, experiment, iteration, self._current_result_dir)
                
                # Training step
                if iteration < self.config.training_iterations:
                    loss = synthetic_generator.train_step(self.signal_precomputer)
                    
                    # Track training history
                    self.training_history['losses'].append(loss)
                    self.training_history['iterations'].append(iteration)
                    self.training_history['timestamps'].append(time.time())
                
                # Save final results
                if iteration == self.config.training_iterations:
                    final_images, final_labels = synthetic_generator.get_evaluation_data()
                    self.saved_data.append([copy.deepcopy(final_images.detach().cpu()), 
                                          copy.deepcopy(final_labels.detach().cpu())])
                    
                    # Save final results with structured organization
                    result_dir = self.save_final_results()
                    print(f'Experiment {experiment} completed. Results saved to: {result_dir}')
                    
                    # Save final visualization in the result directory
                    self.evaluator.save_visualization(final_images, experiment, iteration, result_dir)
        
        # Print final results
        self.evaluator.print_final_results()
    
    def _save_checkpoint(self, experiment: int):
        """Save training checkpoint with structured path."""
        # Create structured result directory
        result_dir = self._create_result_directory()
        
        # Create checkpoint filename with experiment number
        checkpoint_filename = f'experiment_{experiment:02d}_checkpoint.pt'
        checkpoint_path = os.path.join(result_dir, checkpoint_filename)
        
        # Save checkpoint with metadata
        checkpoint_data = {
            'data': self.saved_data,
            'accuracies_all_experiments': self.evaluator.accuracies_all_experiments,
            'config': self.config.__dict__,
            'experiment': experiment,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    
    def _create_result_directory(self) -> str:
        """Create structured result directory with hyperparameters."""
        # Define hyperparameters to include in path
        hp_dict = {
            'dataset': self.config.dataset,
            'model': self.config.model,
            'ipc': self.config.images_per_class,
            'epsilon': self.config.noise_sigma,  # Using noise_sigma as epsilon
            'delta': 1e-5,  # Default delta value for DP
            'arch': self.config.model,  # Architecture
            'method': self.config.method,
            'ser_dim': self.config.ser_dimension,
            'group_size': self.config.group_size,
            'lr_img': self.config.lr_images,
            'lr_net': self.config.lr_net,
            'batch': self.config.batch_train,
            'iterations': self.config.training_iterations,
            'sampling_iter': self.config.sampling_iterations,
            'pea': 'on' if self.config.enable_pea else 'off',
            'ser': 'on' if self.config.enable_ser else 'off',
            'dsa': 'on' if self.config.dsa else 'off'
        }
        
        # Create compact directory name with key hyperparameters
        dir_name_parts = [
            f"{hp_dict['dataset']}",
            f"{hp_dict['model']}",
            f"ipc{hp_dict['ipc']}",
            f"eps{hp_dict['epsilon']:.2f}",
            f"delta{hp_dict['delta']:.0e}",
            f"ser{hp_dict['ser_dim']}",
            f"group{hp_dict['group_size']}",
            f"lr{hp_dict['lr_img']:.2f}",
            f"batch{hp_dict['batch']}",
            f"iter{hp_dict['iterations']}",
            f"pea{hp_dict['pea'][:2]}",  # Just 'on' or 'of'
            f"ser{hp_dict['ser'][:2]}",
            f"dsa{hp_dict['dsa'][:2]}"
        ]
        
        # Create timestamp for uniqueness
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{'_'.join(dir_name_parts)}"
        
        # Create full path
        result_dir = os.path.join(self.config.save_path, dir_name)
        os.makedirs(result_dir, exist_ok=True)
        
        print(f'Created result directory: {result_dir}')
        return result_dir
    
    def save_final_results(self):
        """Save final results and summary."""
        result_dir = self._create_result_directory()
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(result_dir, 'final_checkpoint.pt')
        final_data = {
            'data': self.saved_data,
            'accuracies_all_experiments': self.evaluator.accuracies_all_experiments,
            'training_history': self.training_history,
            'config': self.config.__dict__,
            'final_results': True,
            'timestamp': time.time()
        }
        torch.save(final_data, final_checkpoint_path)
        
        # Save configuration as JSON for easy reading
        import json
        config_path = os.path.join(result_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Save results summary
        summary_path = os.path.join(result_dir, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DOSser Training Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.config.__dict__.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nFinal Results:\n")
            f.write("-" * 20 + "\n")
            for model_key, accuracies in self.evaluator.accuracies_all_experiments.items():
                if accuracies:
                    mean_acc = np.mean(accuracies) * 100
                    std_acc = np.std(accuracies) * 100
                    f.write(f"{model_key}: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        
        # Save training plots if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            self._save_training_plots(result_dir)
        except ImportError:
            print("Matplotlib not available, skipping training plots")
        
        print(f'Final results saved to: {result_dir}')
        return result_dir
    
    def _save_training_plots(self, result_dir: str):
        """Save training loss plots."""
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = os.path.join(result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training loss
        if self.training_history['losses']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_history['iterations'], self.training_history['losses'])
            plt.title('Training Loss Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save loss data as CSV
            import csv
            loss_file = os.path.join(plots_dir, 'training_loss.csv')
            with open(loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'loss', 'timestamp'])
                for i, loss, timestamp in zip(self.training_history['iterations'], 
                                            self.training_history['losses'], 
                                            self.training_history['timestamps']):
                    writer.writerow([i, loss, timestamp])
            
            print(f'Training plots saved to: {plots_dir}')


def parse_arguments() -> TrainingConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description='Synthetic Data Generation and Evaluation for Deep Learning Models')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of the dataset to be used')
    parser.add_argument('--model', type=str, default='ConvNet', help='Model architecture to be used for evaluation')
    parser.add_argument('--ipc', type=int, default=50, help='Number of images per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='Evaluation mode')
    
    # Training parameters
    parser.add_argument('--num_exp', type=int, default=5, help='Number of experimental repetitions')
    parser.add_argument('--num_eval', type=int, default=1, help='Number of evaluations with randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='Number of epochs for training a model with synthetic data')
    parser.add_argument('--sampling_iteration', type=int, default=10000, help='Number of iterations for sampling')
    parser.add_argument('--training_iteration', type=int, default=200000, help='Number of training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='Learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='Learning rate for updating network parameters')
    
    # Algorithm parameters
    parser.add_argument('--PEA', type=bool, default=True, help='Enable or disable Parameter Efficiency Augmentation (PEA)')
    parser.add_argument('--SER', type=bool, default=True, help='Enable or disable Subspace discovery for Error Reduction (SER)')
    parser.add_argument('--aux_path', type=str, default='/data/runkai/PASDA/SD/sd_cifar10_50000_96/', help='Path to auxiliary dataset')
    parser.add_argument('--aux_ipc', type=int, default=1000, help='Number of images per class for the auxiliary dataset')
    parser.add_argument('--ser_dim', type=int, default=2048, help='Dimensionality of the subspace for SER')
    parser.add_argument('--group_size', type=int, default=50, help='Size of the group for each sampling iteration')
    parser.add_argument('--batch_train', type=int, default=256, help='Batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='DSA strategy')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--save_path', type=str, default='result', help='Directory to save the results')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Maximum norm of the representations')
    parser.add_argument('--sigma', type=float, default=4.2969, help='Noise multiplier for added noise in synthetic data')
    
    args = parser.parse_args()
    
    # Convert to TrainingConfig
    config = TrainingConfig(
        dataset=args.dataset,
        model=args.model,
        images_per_class=args.ipc,
        eval_mode=args.eval_mode,
        num_experiments=args.num_exp,
        num_evaluations=args.num_eval,
        epoch_eval_train=args.epoch_eval_train,
        sampling_iterations=args.sampling_iteration,
        training_iterations=args.training_iteration,
        lr_images=args.lr_img,
        lr_net=args.lr_net,
        enable_pea=args.PEA,
        enable_ser=args.SER,
        aux_dataset_path=args.aux_path,
        aux_images_per_class=args.aux_ipc,
        ser_dimension=args.ser_dim,
        group_size=args.group_size,
        batch_train=args.batch_train,
        dsa_strategy=args.dsa_strategy,
        data_path=args.data_path,
        save_path=args.save_path,
        max_norm=args.max_norm,
        noise_sigma=args.sigma
    )
    
    # The __post_init__ method will automatically set device, dsa_param, and dsa
    
    return config


def main():
    """Main function."""
    config = parse_arguments()
    trainer = DosserTrainer(config)
    trainer.run_experiments()


if __name__ == '__main__':
    main()
