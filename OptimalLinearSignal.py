import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from scipy.stats import t
from sklearn.decomposition import PCA
import heapq  

class OptimalLinearSignal:
    def __init__(self, pivot: pd.Series, lambda_reg: float = 1, l1_ratio: float=0.1, 
                 k_principal_components:int=0, 
                 p_value_threshold:float=0.001
                 ) -> None:
        """
        Initialize the model with the given parameters.
        :param pivot: DataFrame containing historical pivot data.
        :param lambda_reg: The overall regularization parameter (non-negative).
        :param l1_reg: Weight for L1 regularization term, should be in [0, 1].
        """

        self.pivot = pivot.copy()  # Copy the pivot dataframe
        self.lambda_l1reg = lambda_reg * l1_ratio  # L1 regularization term
        self.lambda_l2reg = lambda_reg * (1 - l1_ratio)  # L2 regularization term
        self.pca = PCA(n_components=k_principal_components) if k_principal_components > 0 else None # Set a PCA if k_principal_components is specified
        self.p_val_threshold = p_value_threshold # Set threshold for p_value for statistic significance regularization
        
    def __transform__(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input features for model fitting or prediction.
        
        :param X: DataFrame of input features.
        :return: Transformed DataFrame.
        """

        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add an intercept term to the features.
        X_tilde = X_local.shift(1).multiply(self.pivot.loc[X_local.index], axis=0)  # Apply h-operator transformation.

        return X_tilde  
    
    def __apply_pca__(self, X:pd.DataFrame) -> pd.DataFrame:

        # Fit the PCA if it has not been fit yet
        if not hasattr(self.pca, "components_"): self.pca.fit(X.ffill().fillna(0))
        
        # return the pca-transformed df
        return pd.DataFrame(X.dot(self.pca.components_.T), index=X.index)
        # Instead of the standard pca.transform method, direct matrix multiplication with PCA components is used. 
        # This approach is chosen because the standard pca.transform method scales the data before projection, 
        # which is not desired in this specific context. This detail is crucial as scaling can significantly alter 
        # the data characteristics, leading to different results.

    def __get_optimal_beta__(self, mu: np.array, sigma: np.array) -> np.array:
        """
        Conducts optimization to determine the optimal beta coefficients.
        Parameters:
        mu (np.array): The mean vector of the transformed features.
        sigma (np.array): The covariance matrix of the transformed features.
        Returns:
        np.array: Optimal beta coefficients.
        """
        
        if self.lambda_l2reg: # Apply L2 regularization if specified
            sigma += self.lambda_l2reg * (np.linalg.norm(sigma, ord='fro') / len(sigma)) * np.eye(len(sigma))
            sigma /= (1 + self.lambda_l2reg)

        # Verify if Sigma matrix is invertible
        if np.linalg.det(sigma) == 0: raise ValueError("Cov matrix must be invertible, try increasing lambda_reg.")

        # Compute optimal beta 
        sigma_inv = np.linalg.inv(sigma)
        beta_hat = sigma_inv.dot(mu) / np.sqrt(mu.dot(sigma_inv.dot(mu)))

        if self.lambda_l1reg: # Apply L1 regularization if specified
            lambda_l1_true = self.lambda_l1reg * beta_hat.dot(mu)
            loss = lambda beta: - beta.dot(mu) + lambda_l1_true * np.abs(beta).sum()
            grad_loss = lambda beta: (sigma.dot(beta) - mu) + lambda_l1_true * np.sign(beta).sum()
            constraint = {'type': 'eq', 'fun': lambda beta: beta.dot(sigma.dot(beta)) - 1}
            beta_hat = minimize(fun=loss, jac=grad_loss, constraints=constraint, x0=beta_hat, method='SLSQP').x 
        
        return beta_hat  
    
    def __stat_significance_regularization__(self, p_value_threshold:float):
        """
        Applies t-test regularization to adjust the beta coefficients.

        This method focuses on modifying the beta coefficients of a model based on statistical significance determined by a t-test. 
        Beta coefficients whose corresponding p-values exceed a predefined threshold are set to zero, implying they are not statistically 
        significant for the model.

        Note: 
        - The method assumes that 'self.beta', 'self.mu', 'self.tau' are pre-defined in the class.
        - The method mutates 'self.beta', directly modifying the beta coefficients of the class instance.
        """

        #A lambda function 'p_val' is defined to calculate the two-tailed p-value for a t-distribution. 
        #It uses the cumulative distribution function (CDF) of the t-distribution.
        p_val = lambda val, k: 2 * min(t.cdf(val, k), 1 - t.cdf(val, k))

        #The 'beta_test' variable is calculated by normalizing the beta coefficients      
        beta_test =  self.beta * self.beta.dot(self.mu) * np.sqrt(self.tau)

        # The beta coefficients are then iteratively checked against the p-value threshold. 
        # If the p-value for a coefficient (calculated using 'beta_test') is greater than the threshold, 
        # that coefficient is set to zero, indicating it is statistically insignificant
        self.beta = np.array([self.beta[i] if p_val(beta_test_i, self.tau - 1) > p_value_threshold else 0 for i, beta_test_i in enumerate(beta_test)])  

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the model based on the input features X.
        :param X: DataFrame containing feature data with index aligned to pivot data.
        """

        # Store feature names for consistency checks.
        self.features = X.columns  

        # Compute training size 
        self.tau = len(X)

        # Transform features using h-operator
        X_tilde = self.__transform__(X)

        # Apply pca if specificied
        if self.pca: X_tilde = self.__apply_pca__(X_tilde)

        # Compute mean and covariance of transformed features
        self.mu = np.array(X_tilde.mean())
        self.sigma = np.array(X_tilde.cov())

        # Get the optimal beta values
        self.beta = self.__get_optimal_beta__(self.mu, self.sigma)

        #Apply statistical signficance regularization if specified 
        if self.p_val_threshold: self.__stat_significance_regularization__(self.p_val_threshold)

    def predict_optimal_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions of the Optimal Linear Signal on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted signals.
        """

        if set(X.columns) != set(self.features): raise ValueError("The feature names should match those that were passed during fit.")
        
        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept
        
        # Apply PCA that has been fitted on train data
        if self.pca: X_local = self.__apply_pca__(X_local)

        # Compute and return optimal signals
        return X_local.dot(self.beta)    

    def predict_optimal_pnl(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions of the Optimal Linear PnL on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted signals.
        """        
        if set(X.columns) != set(self.features): raise ValueError("The feature names should match those that were passed during fit.")

        X_tilde = self.__transform__(X)
    
        if self.pca: X_tilde = self.__apply_pca__(X_tilde)

        return X_tilde.dot(self.beta)   
        
    ##########    ##########    ##########    ##########    ##########
    ##########    ########  Additionnal methods ########    ##########
    ##########    ##########    ##########    ##########    ##########

    def get_weight(self) -> np.array:
        """
        Retrieve the optimized weight vector.
        :return: Optimized beta values.
        """
        return self.beta  # Return the optimized beta values
    
    def get_k_best(self, k) -> list[str]:
        """
        Identify the 'k' largest absolute values in the optimized beta vector and return their indices.
        :param k: The number of largest elements to identify.
        :return: List of indices corresponding to the 'k' largest absolute values in the beta vector.
        """

        if self.pca: raise ValueError("The method get_k_best is not applicable when PCA is applied, as PCA already performs feature transformation and reduction.")

        # Use heapq.nlargest to get the 'k' largest absolute values. 
        k_largest_elements = heapq.nlargest(k, enumerate(self.beta), key=lambda x: np.abs(x[1]))
        
        # Extract the indices of the elements from k_largest_elements.
        indexs_k_bests = [index for index, _ in k_largest_elements if index<len(self.features)]

        return self.features[indexs_k_bests]  # Return the list of features corresponding to indices.
    
    def evaluate(self)->float: 
        """
        Evaluate the quality of the model based on the optimized beta values.
        """
        obj_funct = lambda beta, mu, sigma : beta.dot(mu) / np.sqrt(beta.dot(sigma.dot(beta)))

        return obj_funct(self.beta, self.mu, self.sigma)  # Evaluate the model based on the objective function
    
    