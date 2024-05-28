package com.example.machinelearning.classification;

public interface ClassificationModel {
    // Interface for classification models

    /**
     * Train the classification model on the given training data.
     * 
     * @param X Input features matrix of shape (n_samples, n_features).
     * @param y Array of target labels of shape (n_samples).
     */
    void fit(double[][] X, int[] y);

    /**
     * Predict the target labels for the input features.
     * 
     * @param X Input features matrix of shape (n_samples, n_features).
     * @return Array of predicted labels of shape (n_samples).
     */
    int[] predict(double[][] X);
}
