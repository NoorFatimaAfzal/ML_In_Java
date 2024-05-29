package com.example.machinelearning.classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNN {

    private final int k;
    private final List<double[]> X;
    private final List<Integer> y;

    public KNN(int k) {
        this.k = k;
        this.X = new ArrayList<>();
        this.y = new ArrayList<>();
    }

    public void fit(double[][] X, int[] y) {
        for (int i = 0; i < X.length; i++) {
            this.X.add(X[i]);
            this.y.add(y[i]);
        }
    }

    public int predict(double[] x) {
        // Compute the Euclidean distance of x from each sample in the training dataset
        List<double[]> distances = new ArrayList<>();
        for (int i = 0; i < this.X.size(); i++) {
            double distance = euclideanDistance(x, this.X.get(i));
            distances.add(new double[]{distance, this.y.get(i)});
        }

        // Sort according to the distances computed and extract the nearest k neighbors
        Collections.sort(distances, Comparator.comparingDouble(o -> o[0]));

        // Print distances and class labels of k-nearest neighbors
        System.out.println("Distances and Class labels of " + k + "-nearest neighbors are:");
        for (int i = 0; i < k; i++) {
            System.out.println("Distance: " + distances.get(i)[0] + ", Class: " + (int) distances.get(i)[1]);
        }

        // Extract the class labels of the k nearest neighbors
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            labels.add((int) distances.get(i)[1]);
        }

        // Find the most frequently occurring element
        int mostCommonLabel = mode(labels);
        return mostCommonLabel;
    }

    private double euclideanDistance(double[] x1, double[] x2) {
        double sumOfSquares = 0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sumOfSquares += diff * diff;
        }
        return Math.sqrt(sumOfSquares);
    }

    private int mode(List<Integer> labels) {
        Map<Integer, Integer> labelCounts = new HashMap<>();
        for (int label : labels) {
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }
        return Collections.max(labelCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
      
}
