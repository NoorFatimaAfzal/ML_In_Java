package com.example.machinelearning.regression;

public class SimpleLinearRegression implements RegressionModel {

    private double slope;
    private double intercept;

    public SimpleLinearRegression() {
        this.slope = 0;
        this.intercept = 0;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        // Assuming X is a single feature for simplicity (n_samples, 1)
        int n = X.length;

        if (X[0].length != 1) {
            throw new IllegalArgumentException("SimpleLinearRegression only supports single feature inputs.");
        }

        double sumX = 0.0, sumY = 0.0;
        for (int i = 0; i < n; i++) {
            sumX += X[i][0];
            sumY += y[i];
        }

        double xMean = sumX / n;
        double yMean = sumY / n;

        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < n; i++) {
            double xDiff = X[i][0] - xMean;
            double yDiff = y[i] - yMean;

            numerator += xDiff * yDiff;
            denominator += xDiff * xDiff;
        }

        slope = numerator / denominator;
        intercept = yMean - slope * xMean;
    }

    @Override
    public double[] predict(double[][] X) {
        // Assuming X is a single feature for simplicity (n_samples, 1)
        int n = X.length;
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = slope * X[i][0] + intercept;
        }

        return predictions;
    }

    public double getSlope() {
        return slope;
    }

    public double getIntercept() {
        return intercept;
    }

    // public static void main(String[] args) {
    //     double[][] X = { {1}, {2}, {3}, {4}, {5} };
    //     double[] y = { 4, 3, 5, 5, 6 };

    //     SimpleLinearRegression model = new SimpleLinearRegression();
    //     model.fit(X, y);

    //     System.out.println("Slope is: " + model.getSlope());
    //     System.out.println("Intercept is: " + model.getIntercept());

    //     double[][] testX = { {6} };
    //     double[] predictions = model.predict(testX);
    //     System.out.println("Predicted y for x = 6: " + predictions[0]);
    // }
}
