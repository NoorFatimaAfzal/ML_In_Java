package com.example.machinelearning.utils;

public class Metrics {

    // Mean Absolute Error (MAE) for regression
    public static double meanAbsoluteError(double[] trueValues, double[] predictedValues) {
        double errorSum = 0;
        for (int i = 0; i < trueValues.length; i++) {
            errorSum += Math.abs(trueValues[i] - predictedValues[i]);
        }
        return errorSum / trueValues.length;
    }

    // Mean Squared Error (MSE) for regression
    public static double meanSquaredError(double[] trueValues, double[] predictedValues) {
        double errorSum = 0;
        for (int i = 0; i < trueValues.length; i++) {
            errorSum += Math.pow(trueValues[i] - predictedValues[i], 2);
        }
        return errorSum / trueValues.length;
    }

    // Root Mean Squared Error (RMSE) for regression
    public static double rootMeanSquaredError(double[] trueValues, double[] predictedValues) {
        return Math.sqrt(meanSquaredError(trueValues, predictedValues));
    }

    // Accuracy for classification
    public static double accuracy(int[] trueLabels, int[] predictedLabels) {
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == predictedLabels[i]) {
                correct++;
            }
        }
        return (double) correct / trueLabels.length;
    }

    // Precision for binary classification
    public static double precision(int[] trueLabels, int[] predictedLabels, int positiveClass) {
        int truePositives = 0;
        int predictedPositives = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (predictedLabels[i] == positiveClass) {
                predictedPositives++;
                if (trueLabels[i] == positiveClass) {
                    truePositives++;
                }
            }
        }
        if (predictedPositives == 0) {
            return 0;
        }
        return (double) truePositives / predictedPositives;
    }

    // Recall for binary classification
    public static double recall(int[] trueLabels, int[] predictedLabels, int positiveClass) {
        int truePositives = 0;
        int actualPositives = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == positiveClass) {
                actualPositives++;
                if (predictedLabels[i] == positiveClass) {
                    truePositives++;
                }
            }
        }
        if (actualPositives == 0) {
            return 0;
        }
        return (double) truePositives / actualPositives;
    }

    // F1 Score for binary classification
    public static double f1Score(int[] trueLabels, int[] predictedLabels, int positiveClass) {
        double precision = precision(trueLabels, predictedLabels, positiveClass);
        double recall = recall(trueLabels, predictedLabels, positiveClass);
        if (precision + recall == 0) {
            return 0;
        }
        return 2 * (precision * recall) / (precision + recall);
    }

    // Method to generate a confusion matrix
    public static int[][] confusionMatrix(double[] yTrue, double[] yPred, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];

        // Populate the confusion matrix
        for (int i = 0; i < yTrue.length; i++) {
            int trueClass = (int) yTrue[i];
            int predClass = (int) yPred[i];
            matrix[trueClass][predClass]++;
        }

        return matrix;
    }
}
