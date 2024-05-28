package com.example.machinelearning.regression;

public class PolynomialRegression implements RegressionModel {
    private final int degree;
    private double[] coefficients;

    public PolynomialRegression(int degree) {
        this.degree = degree;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        double[][] polyFeatures = createPolynomialFeatures(X, degree);

        int n = polyFeatures.length;
        int m = polyFeatures[0].length;

        double[][] X_transpose = new double[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X_transpose[j][i] = polyFeatures[i][j];
            }
        }

        double[][] XTX = multiplyMatrices(X_transpose, polyFeatures);
        double[][] XTX_inv = invertMatrix(XTX);
        double[][] XTY = multiplyMatrixVector(X_transpose, y);

        coefficients = new double[m];
        for (int i = 0; i < m; i++) {
            coefficients[i] = 0;
            for (int j = 0; j < m; j++) {
                coefficients[i] += XTX_inv[i][j] * XTY[j][0];
            }
        }
    }

    @Override
    public double[] predict(double[][] X) {
        double[][] polyFeatures = createPolynomialFeatures(X, degree);

        int n = polyFeatures.length;
        int m = polyFeatures[0].length;
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = 0;
            for (int j = 0; j < m; j++) {
                predictions[i] += coefficients[j] * polyFeatures[i][j];
            }
        }

        return predictions;
    }

    private double[][] createPolynomialFeatures(double[][] X, int degree) {
        int n = X.length; //number of samples
        int m = X[0].length; //number of features
        //(degree+1)^m - 1 features
        // -1 to exclude the constant term
        int numFeatures = (int) Math.pow(degree + 1, m) - 1;
        double[][] polyFeatures = new double[n][numFeatures];

        // powers calculate krny k liye 
        
        for (int i = 0; i < n; i++) {
            int index = 0;
            for (int d = 0; d <= degree; d++) {
                for (int j = 0; j < m; j++) {
                    polyFeatures[i][index++] = Math.pow(X[i][j], d);
                }
            }
        }

        return polyFeatures;
    }

    // Helper method to multiply two matrices
    private double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        int r1 = firstMatrix.length;
        int c1 = firstMatrix[0].length;
        int c2 = secondMatrix[0].length;
        double[][] result = new double[r1][c2];

        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                result[i][j] = 0;
                for (int k = 0; k < c1; k++) {
                    result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
                }
            }
        }

        return result;
    }

    // Helper method to multiply a matrix and a vector
    private double[][] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][1];

        for (int i = 0; i < rows; i++) {
            result[i][0] = 0;
            for (int j = 0; j < cols; j++) {
                result[i][0] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    // Helper method to invert a matrix
    private double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmentedMatrix = new double[n][2 * n];

        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, augmentedMatrix[i], 0, n);
            augmentedMatrix[i][i + n] = 1;
        }

        for (int i = 0; i < n; i++) {
            double maxElement = Math.abs(augmentedMatrix[i][i]);
            int maxRow = i;

            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmentedMatrix[k][i]) > maxElement) {
                    maxElement = Math.abs(augmentedMatrix[k][i]);
                    maxRow = k;
                }
            }

            double[] temp = augmentedMatrix[maxRow];
            augmentedMatrix[maxRow] = augmentedMatrix[i];
            augmentedMatrix[i] = temp;

            for (int k = i + 1; k < 2 * n; k++) {
                augmentedMatrix[i][k] /= augmentedMatrix[i][i];
            }

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmentedMatrix[k][i];
                    for (int j = i; j < 2 * n; j++) {
                        augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
                    }
                }
            }
        }

        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmentedMatrix[i][j + n];
            }
        }

        return inverse;
    }

    // public static void main(String[] args) {
    //     double[][] data = {
    //         {1, 2},
    //         {2, 3},
    //         {4, 5},
    //         {6, 7}
    //     };
    //     double[] target = {2, 3, 4, 5};

    //     PolynomialRegression model = new PolynomialRegression(2);

    //     model.fit(data, target);

    //     double[][] newData = {
    //         {3, 4},
    //         {5, 6}
    //     };
    //     double[] predictions = model.predict(newData);
    //     System.out.println("Predictions: " + Arrays.toString(predictions));
    // }
}
