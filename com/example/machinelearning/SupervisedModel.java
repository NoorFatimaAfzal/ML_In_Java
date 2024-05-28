package com.example.machinelearning;

// Abstract class SupervisedModel extends Model
public abstract class SupervisedModel extends Model {
    // Abstract method for evaluating the model
    public abstract double evaluate(double[][] X, double[] y);
}
