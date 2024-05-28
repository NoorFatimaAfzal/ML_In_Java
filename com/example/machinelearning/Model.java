package com.example.machinelearning;

public abstract class Model {
    // Abstract methods for fitting the model and making predictions
    public abstract void fit(double[][] X, double[] y);
    public abstract double[] predict(double[][] X);
}
