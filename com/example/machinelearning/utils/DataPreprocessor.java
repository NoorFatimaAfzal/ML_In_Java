package com.example.machinelearning.utils;

import java.util.Arrays;

public class DataPreprocessor {

    // Normalize the data using min-max normalization

    public static double[] minMaxNormalize(double[] data) {
        // Find the minimum and maximum values in the data
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double value : data) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
        
        // Perform min-max normalization
        double[] normalizedData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normalizedData[i] = (data[i] - min) / (max - min);
        }
        
        return normalizedData;
    }

    // Fill missing values with column mean
    public static double[][] fillMissingValuesWithMean(double[][] data) {
        double[][] filledData = new double[data.length][data[0].length];
        double[] means = new double[data[0].length];
        Arrays.fill(means, 0);
        int[] counts = new int[data[0].length];

        // Calculate column means
        for (double[] data1 : data) {
            for (int j = 0; j < data[0].length; j++) {
                if (!Double.isNaN(data1[j])) {
                    means[j] += data1[j];
                    counts[j]++;
                }
            }
        }
        for (int j = 0; j < means.length; j++) {
            if (counts[j] > 0) {
                means[j] /= counts[j];
            }
        }

        // Fill missing values with column means
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                if (Double.isNaN(data[i][j])) {
                    filledData[i][j] = means[j];
                } else {
                    filledData[i][j] = data[i][j];
                }
            }
        }
        return filledData;
    }

}
