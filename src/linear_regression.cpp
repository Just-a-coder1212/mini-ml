
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

struct Dataset {
    vector<float> X;  // Contiguous row-major feature matrix
    vector<float> y;
    size_t rows = 0;
    size_t cols = 0;
};

Dataset load_csv(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open CSV file: " + filename);
    }

    Dataset data;
    string line;

    // Read header and count columns.
    if (!getline(file, line)) {
        throw runtime_error("CSV file is empty.");
    }

    data.cols = 1;
    for (char c : line) {
        if (c == ',') ++data.cols;
    }

    if (data.cols < 2) {
        throw runtime_error("CSV must contain features and a target.");
    }

    const size_t n_features = data.cols - 1;

    // Parse numeric CSV rows.
    while (getline(file, line)) {
        if (line.empty()) continue;

        const char* p = line.c_str();
        const char* end = p + line.size();

        vector<float> row;
        row.reserve(data.cols);

        while (p < end) {
            char* next = nullptr;
            float value = strtof(p, &next);

            if (next == p) {
                throw runtime_error("Invalid numeric value in CSV.");
            }

            row.push_back(value);
            p = next;

            if (p < end && *p == ',') ++p;
            else if (p < end) {
                throw runtime_error("Invalid CSV delimiter.");
            }
        }

        if (row.size() != data.cols) {
            throw runtime_error("Inconsistent number of CSV columns.");
        }

        for (size_t j = 0; j < n_features; ++j) {
            data.X.push_back(row[j]);
        }

        data.y.push_back(row.back());
        ++data.rows;
    }

    if (data.rows == 0) {
        throw runtime_error("CSV contains no data rows.");
    }

    return data;
}

int main() {
    try {
        // Loading is deliberately outside the training timer.
        Dataset data = load_csv("linear_regression_1m.csv");

        const size_t N = data.rows;
        const size_t d = data.cols - 1;

        // 80:20 split, without shuffling.
        const size_t train_size =
            static_cast<size_t>(round(N * 0.80));

        const size_t test_size = N - train_size;

        // Zero-initialized feature weights and intercept.
        vector<double> weights(d, 0.0);
        double bias = 0.0;

        const double lr = 0.001;
        const int epochs = 1000;
        const double inv_m = 1.0 / train_size;

        // Allocate gradients once; reuse every epoch.
        vector<double> gradients(d, 0.0);
        double bias_gradient = 0.0;

        cout << "Rows: " << N
             << ", Features: " << d << '\n';
        cout << "Training samples: " << train_size
             << ", Test samples: " << test_size << '\n';

        const auto start = Clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            fill(gradients.begin(), gradients.end(), 0.0);
            bias_gradient = 0.0;

            // Accumulate batch gradients over training rows.
            for (size_t i = 0; i < train_size; ++i) {
                const float* row = data.X.data() + i * d;

                double prediction = bias;

                for (size_t j = 0; j < d; ++j) {
                    prediction += weights[j] * row[j];
                }

                const double error =
                    prediction - static_cast<double>(data.y[i]);

                bias_gradient += error;

                for (size_t j = 0; j < d; ++j) {
                    gradients[j] += error * row[j];
                }
            }

            // Update each parameter once per epoch.
            for (size_t j = 0; j < d; ++j) {
                weights[j] -= lr * gradients[j] * inv_m;
            }

            bias -= lr * bias_gradient * inv_m;
        }

        const auto end = Clock::now();

        const double seconds =
            chrono::duration<double>(end - start).count();

        // Evaluate on the held-out test set.
        double squared_error_sum = 0.0;

        for (size_t i = train_size; i < N; ++i) {
            const float* row = data.X.data() + i * d;
            double prediction = bias;

            for (size_t j = 0; j < d; ++j) {
                prediction += weights[j] * row[j];
            }

            const double error =
                prediction - static_cast<double>(data.y[i]);

            squared_error_sum += error * error;
        }

        const double test_mse = squared_error_sum / test_size;

        cout << fixed << setprecision(6);
        cout << "\nTraining finished!\n";
        cout << "Training time: " << seconds << " seconds\n";
        cout << "Test MSE: " << test_mse << '\n';
        cout << "Intercept: " << bias << '\n';

        cout << "Weights:\n";
        for (size_t j = 0; j < d; ++j) {
            cout << "w" << j + 1 << " = " << weights[j] << '\n';
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
