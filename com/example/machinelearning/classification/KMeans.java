package com.example.machinelearning.classification;

import java.util.Arrays;
import java.util.Random;

public class KMeans {

    private final int k;  // number of clusters
    private final int maxIterations;  // maximum number of iterations
    private double[][] centroids;  // array of centroid points

    public KMeans(int k, int maxIterations) {
        this.k = k;
        this.maxIterations = maxIterations;
    }

    public void fit(double[][] data) {
        int nSamples = data.length;
        int nFeatures = data[0].length;

        // Initialize centroids
        centroids = new double[k][nFeatures];
        Random random = new Random();

        for (int i = 0; i < k; i++) {
            centroids[i] = data[random.nextInt(nSamples)];
        }

        int[] labels = new int[nSamples];
        boolean centroidsChanged;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            centroidsChanged = false;

            // Assign clusters
            for (int i = 0; i < nSamples; i++) {
                labels[i] = nearestCentroid(data[i]);
            }

            // Update centroids
            double[][] newCentroids = new double[k][nFeatures];
            int[] counts = new int[k];

            for (int i = 0; i < nSamples; i++) {
                int cluster = labels[i];
                for (int j = 0; j < nFeatures; j++) {
                    newCentroids[cluster][j] += data[i][j];
                }
                counts[cluster]++;
            }

            for (int i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    for (int j = 0; j < nFeatures; j++) {
                        newCentroids[i][j] /= counts[i];
                    }
                }
            }

            // Check for convergence
            for (int i = 0; i < k; i++) {
                if (!Arrays.equals(centroids[i], newCentroids[i])) {
                    centroidsChanged = true;
                    break;
                }
            }

            centroids = newCentroids;

            if (!centroidsChanged) {
                break;
            }
        }
    }

    public int[] predict(double[][] data) {
        int[] labels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            labels[i] = nearestCentroid(data[i]);
        }
        return labels;
    }

    private int nearestCentroid(double[] sample) {
        double minDistance = Double.MAX_VALUE;
        int nearest = -1;
        for (int i = 0; i < centroids.length; i++) {
            double distance = euclideanDistance(sample, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearest = i;
            }
        }
        return nearest;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    // public static void main(String[] args) {
    //     double[][] data = {
    //         {1.0, 2.0},
    //         {1.5, 1.8},
    //         {5.0, 8.0},
    //         {8.0, 8.0},
    //         {1.0, 0.6},
    //         {9.0, 11.0},
    //         {8.0, 2.0},
    //         {10.0, 2.0},
    //         {9.0, 3.0}
    //     };

    //     KMeans kMeans = new KMeans(3, 100);
    //     kMeans.fit(data);

    //     System.out.println("Centroids:");
    //     for (double[] centroid : kMeans.centroids) {
    //         System.out.println(Arrays.toString(centroid));
    //     }

    //     int[] labels = kMeans.predict(data);
    //     System.out.println("Labels:");
    //     System.out.println(Arrays.toString(labels));
    // }
}