# Machine Learning Labs

A collection of 5 foundational machine learning assignments implemented from scratch in Python.

---

## Task 1: Principal Component Analysis (PCA) & k-Nearest Neighbors (kNN)

### 1. Problem
Implement Principal Component Analysis (`my_PCA`) and a k-Nearest Neighbors classifier (`my_kNN`) from scratch using NumPy to recognize handwritten digits ($28 \times 28$ pixel images) from the **MNIST** dataset. Evaluate how dimensionality reduction impacts classification accuracy and feature separability.

### 2. Approach
* **PCA (`my_PCA`):** Projects high-dimensional image data ($D = 784$) onto an orthogonal subspace of maximum variance, reducing noise and computational cost:
  1. Center the data: $X_{\text{centered}} = X - \mu$, where $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$
  2. Compute the covariance matrix: $C = X_{\text{centered}}^T X_{\text{centered}}$
  3. Perform eigendecomposition $C F = F \Lambda$ and sort eigenvectors $F$ by descending eigenvalues $\lambda_i$.
  4. Project data onto the top $k$ eigenvectors: $Y = X_{\text{centered}} F_k$
* **kNN (`my_kNN`):** A non-parametric metric classifier that assigns each test sample to the majority class among its $k$ closest training neighbors using Euclidean distance: $d(x, x_i) = \|x - x_i\|_2$.

### 3. Results
* **Variance Retention:** The first **15** principal components capture **~60%** of the total dataset variance (visualized via eigenvalue distribution and cumulative explained variance plots).
* **2D Separability:** Projecting MNIST onto the top 2 principal components reveals substantial class overlap, confirming that digit classes are not linearly separable in 2D.
* **Classification Accuracy:** Grid search across `n_components` $\in [1, 64]$ and `n_neighbors` $\in [1, 30]$ demonstrates that PCA dimensionality reduction preserves or improves kNN accuracy while accelerating distance computations.

### 4. How to Run
1. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn

### 5. My Contribution
1. Implemented core machine learning algorithms (my_PCA, my_kNN, and subsequent lab models) from scratch in Python/NumPy without relying on high-level scikit-learn estimators.
2. Conducted exploratory data analysis, dimensionality reduction experiments, and hyperparameter grid searches.
3. Built reproducible visualizations.
