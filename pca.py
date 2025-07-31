import torch

def perform_pca_lowrank(X, num_components):
    """
    Perform PCA on the input tensor X using torch.pca_lowrank.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features).
        num_components (int): Number of principal components to retain.

    Returns:
        principal_components (torch.Tensor): Principal components of shape (n_features, num_components).
        X_mean (torch.Tensor): Mean of the original data of shape (n_features,).
    """
    # Perform PCA
    U, S, V = torch.pca_lowrank(X, q=num_components, center=True)
    
    # V contains the principal components
    principal_components = V  # Shape: (n_features, num_components)
    
    # Compute the mean of X
    X_mean = torch.mean(X, dim=0)  # Shape: (n_features,)
    
    return principal_components, X_mean

def project_new_data(A, principal_components, X_mean):
    """
    Project new data A onto the principal components obtained from X.

    Args:
        A (torch.Tensor): New data tensor of shape (m_samples, n_features).
        principal_components (torch.Tensor): Principal components from PCA of X, shape (n_features, num_components).
        X_mean (torch.Tensor): Mean of the original data X, shape (n_features,).

    Returns:
        A_pca (torch.Tensor): Projected data of shape (m_samples, num_components).
    """
    # Center the new data using X's mean
    A_centered = A - X_mean  # Shape: (m_samples, n_features)
    
    # Project the centered data onto the principal components
    A_pca = torch.matmul(A_centered, principal_components)  # Shape: (m_samples, num_components)
    
    return A_pca

# Example Usage
if __name__ == "__main__":
    # Example tensor X; replace with your actual data
    X = torch.randn(2000, 2048)
    
    # Number of principal components to keep
    k = 100
    
    # Perform PCA on X
    principal_components, X_mean = perform_pca_lowrank(X, k)
    
    print(f"Principal components shape: {principal_components.shape}")  # Should be (2048, 100)
    print(f"Mean shape: {X_mean.shape}")  # Should be (2048,)
    
    # Project X onto PCA space
    X_pca = project_new_data(X, principal_components, X_mean)
    print(f"Projected X shape: {X_pca.shape}")  # Should be (2000, 100)
    
    # Example new data A; replace with your actual new data
    A = torch.randn(500, 2048)  # For example, 500 new samples
    
    # Project new data A onto PCA space based on X's PCA
    A_pca = project_new_data(A, principal_components, X_mean)
    print(f"Projected A shape: {A_pca.shape}")  # Should be (500, 100)