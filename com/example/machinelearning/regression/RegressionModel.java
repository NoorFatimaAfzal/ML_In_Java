package com.example.machinelearning.regression;

public interface RegressionModel {
    // Interface for regression models

    /**
     * Train the regression model on the given training data.
     * 
     * @param X Input features matrix of shape (n_samples, n_features).
     * @param y Array of target values of shape (n_samples).
     */
    void fit(double[][] X, double[] y);

    /**
     * Predict the target values for the input features.
     * 
     * @param X Input features matrix of shape (n_samples, n_features).
     * @return Array of predicted target values of shape (n_samples).
     */
    double[] predict(double[][] X);
}
