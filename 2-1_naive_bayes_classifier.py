"""
Naive Bayes Classifier for MNIST Handwritten Digit Recognition

This program implements Naive Bayes classifier with two modes:
1. Discrete Mode: Using 32 bins to discretize pixel values (0-255)
2. Continuous Mode: Using Gaussian distribution to model pixel values

Mathematical Foundation:
    Bayes' Theorem: P(Y|X) ∝ P(Y) * ∏ P(Xi|Y)
    - P(Y): Prior probability of class Y
    - P(Xi|Y): Likelihood of feature Xi given class Y
    - All computations in log-space to avoid underflow

Author: 313553058 Ian Tsai
Course: Machine Learning HW2 (2025 Spring)
"""

import sys
import math
import binascii
import numpy as np
import os


def LoadImage(file):
    """
    Load MNIST image file in binary format.

    Args:
        file (str): Filename of MNIST image file (e.g., 'train-images.idx3-ubyte_')

    Returns:
        file object: Opened binary file object positioned after magic number

    File Format (Big Endian):
        [0-3]:   Magic number (2051 for images)
        [4-7]:   Number of images
        [8-11]:  Number of rows (28)
        [12-15]: Number of columns (28)
        [16+]:   Pixel data (0-255 grayscale)
    """
    print(f"[INFO] Loading image file: {file}")
    image_file = open(f'./{file}', 'rb')  # Open in binary read mode
    image_file.read(4)  # Skip magic number (0x00000803 = 2051)
    return image_file


def LoadLabel(file):
    """
    Load MNIST label file in binary format.

    Args:
        file (str): Filename of MNIST label file (e.g., 'train-labels.idx1-ubyte_')

    Returns:
        file object: Opened binary file object positioned after magic number

    File Format (Big Endian):
        [0-3]:  Magic number (2049 for labels)
        [4-7]:  Number of labels
        [8+]:   Label data (0-9)
    """
    print(f"[INFO] Loading label file: {file}")
    label_file = open(f'./{file}', 'rb')  # Open in binary read mode
    label_file.read(4)  # Skip magic number (0x00000801 = 2049)
    return label_file


def PrintResult(prob, answer):
    """
    Print posterior probabilities and prediction result.

    Args:
        prob (np.array): Log-scale posterior probabilities for each class (shape: [10])
                        In Naive Bayes: log P(Y=j|X) for j=0,1,...,9
        answer (int): Ground truth label (0-9)

    Returns:
        int: 0 if prediction is correct, 1 if incorrect (for error counting)

    Mathematical Formula:
        prediction = argmax_j P(Y=j|X)
                   = argmax_j [log P(Y=j) + Σ log P(Xi|Y=j)]
    """
    print('Posterior (in log scale):')
    # Display log-probability for each digit class
    for i in range(prob.shape[0]):
        print(f'{i}: {prob[i]}')

    # Find class with maximum posterior probability
    # Note: prob contains log probabilities (negative values)
    # argmin finds the least negative = maximum probability
    pred = np.argmin(prob)
    print(f'Prediction: {pred}, Ans: {answer}\n')

    # Return 0 for correct prediction, 1 for error
    return 0 if answer == pred else 1


def DrawImagination(image, image_row, image_col, mode):
    """
    Visualize the learned model as binary images for each digit.

    Args:
        image: Learned parameters from training
               - Discrete mode: 3D array [class][pixel][bin] containing bin counts
               - Continuous mode: 2D array [class][pixel] containing mean values
        image_row (int): Number of rows in image (28)
        image_col (int): Number of columns in image (28)
        mode (int): 0 for Discrete mode, 1 for Continuous mode

    Returns:
        None (prints to stdout)

    Output:
        For each digit (0-9), print a 28x28 binary image where:
        - 0 = white pixel (Bayesian model expects low intensity)
        - 1 = black pixel (Bayesian model expects high intensity)
    """
    print('Imagination of numbers in Bayesian classifier\n')

    if mode == 0:  # Discrete Mode
        # For each digit class
        for i in range(10):
            print(f'{i}:')
            # For each row
            for j in range(image_row):
                # For each column
                for k in range(image_col):
                    pixel_idx = j * image_row + k
                    # Sum counts in bins 0-15 (low intensity = white)
                    white = sum(image[i][pixel_idx][:16])
                    # Sum counts in bins 16-31 (high intensity = black)
                    black = sum(image[i][pixel_idx][16:])
                    # Print 1 if black is more common, 0 otherwise
                    print(f'{1 if black > white else 0} ', end='')
                print()  # New line after each row
            print()  # Empty line between digits

    elif mode == 1:  # Continuous Mode
        # For each digit class
        for i in range(10):
            print(f'{i}:')
            # For each row
            for j in range(image_row):
                # For each column
                for k in range(image_col):
                    pixel_idx = j * image_row + k
                    # mean value: if > 128 (middle of 0-255), consider it black
                    print(f'{1 if image[i][pixel_idx] > 128 else 0} ', end='')
                print()  # New line after each row
            print()  # Empty line between digits


def LoadTrainingData():
    """
    Load training images and labels from MNIST dataset.

    Returns:
        tuple: (train_image_file, train_label_file, train_size, image_row, image_col)
            - train_image_file: File object for training images
            - train_label_file: File object for training labels
            - train_size (int): Number of training samples (60000)
            - image_row (int): Image height in pixels (28)
            - image_col (int): Image width in pixels (28)
    """
    print("[INFO] Loading training data...")

    # Load training images
    train_image_file = LoadImage('train-images.idx3-ubyte_')

    # Read header information (Big Endian format)
    train_size = int(binascii.b2a_hex(train_image_file.read(4)), 16)  # Number of images
    image_row = int(binascii.b2a_hex(train_image_file.read(4)), 16)   # Rows (28)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)), 16)   # Columns (28)

    # Load training labels
    train_label_file = LoadLabel('train-labels.idx1-ubyte_')
    train_label_file.read(4)  # Skip number of labels (same as train_size)

    print(f"[INFO] Training size: {train_size}, Image size: {image_row}x{image_col}")
    return train_image_file, train_label_file, train_size, image_row, image_col


def LoadTestingData():
    """
    Load testing images and labels from MNIST dataset.

    Returns:
        tuple: (test_image_file, test_label_file, test_size)
            - test_image_file: File object for testing images
            - test_label_file: File object for testing labels
            - test_size (int): Number of test samples (10000)
    """
    print("[INFO] Loading testing data...")

    # Load test images
    test_image_file = LoadImage('t10k-images.idx3-ubyte_')
    test_size = int(binascii.b2a_hex(test_image_file.read(4)), 16)  # Number of test images
    test_image_file.read(4)  # Skip rows
    test_image_file.read(4)  # Skip columns

    # Load test labels
    test_label_file = LoadLabel('t10k-labels.idx1-ubyte_')
    test_label_file.read(4)  # Skip number of labels

    print(f"[INFO] Test size: {test_size}")
    return test_image_file, test_label_file, test_size


def DiscreteMode(output_file=None):
    """
    Train and evaluate Naive Bayes classifier using Discrete mode.

    Discrete Mode discretizes pixel values (0-255) into 32 bins, then counts
    the frequency of each bin for each (class, pixel) combination.

    Args:
        output_file (str, optional): If provided, save results to this file

    Returns:
        None

    Mathematical Model:
        Prior: P(Y=j) = count(Y=j) / N
        Likelihood: P(Xi=bin_k | Y=j) = count(Y=j, Xi=bin_k) / count(Y=j)
        Posterior: log P(Y=j|X) = log P(Y=j) + Σ log P(Xi|Y=j)

        Bin mapping: bin_index = grayscale_value // 8
            - Bin 0: grayscale 0-7
            - Bin 1: grayscale 8-15
            - ...
            - Bin 31: grayscale 248-255

    Data Structures:
        image[class][pixel][bin]: Count of training samples with class=j,
                                  pixel position=k, in bin=b
        image_sum[class][pixel]: Total count for normalization
        prior[class]: Number of training samples for each class
    """
    print("[INFO] Starting Discrete Naive Bayes training and evaluation...")

    # Load data
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    # ========== Initialize data structures ==========
    image_size = image_row * image_col  # 28 * 28 = 784 pixels

    # image[class][pixel][bin]: frequency count
    # Shape: [10 classes, 784 pixels, 32 bins]
    image = np.zeros((10, image_size, 32), dtype=np.int32)

    # image_sum[class][pixel]: total count for each (class, pixel)
    # Used for normalization: P(Xi=bin|Y) = image[Y][i][bin] / image_sum[Y][i]
    image_sum = np.zeros((10, image_size), dtype=np.int32)

    # prior[class]: count of each class in training data
    # P(Y=j) = prior[j] / train_size
    prior = np.zeros((10), dtype=np.int32)

    output = []  # Store output for file writing

    # ========== Training Phase ==========
    print("[INFO] Training: Building frequency tables...")
    for i in range(train_size):
        if i % 1000 == 0:
            print(f"[TRAIN] Processing image {i}/{train_size}")

        # Read label (0-9) from binary file
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)

        # Update prior count: P(Y = label)
        prior[label] += 1

        # Read all 784 pixels
        for j in range(image_size):
            # Read grayscale value (0-255)
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)

            # Convert grayscale to bin index (0-31)
            # Mathematical mapping: bin = floor(grayscale / 8)
            bin_idx = grayscale // 8

            # Update frequency count
            image[label][j][bin_idx] += 1  # Count this specific bin
            image_sum[label][j] += 1        # Count total for normalization

    # ========== Testing Phase ==========
    print("[INFO] Starting testing...")
    error = 0  # Count prediction errors

    for i in range(test_size):
        if i % 100 == 0:
            print(f"[TEST] Evaluating test image {i}/{test_size}")

        # Read ground truth label
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)

        # Initialize log-posterior probabilities for all 10 classes
        # Mathematical: log P(Y=j|X) for j=0,1,...,9
        prob = np.zeros((10), dtype=float)

        # Read test image (784 pixels)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)

        # ========== Calculate Posterior for each class ==========
        for j in range(10):  # For each class j = 0, 1, ..., 9

            # Add log prior: log P(Y=j)
            prob[j] += np.log(prior[j] / train_size)

            # Add log likelihood for each pixel: Σ log P(Xi|Y=j)
            for k in range(image_size):  # For each pixel position k

                # Find which bin this test pixel falls into
                bin_idx = test_image[k] // 8

                # Get frequency count: image[class=j][pixel=k][bin]
                likelihood = image[j][k][bin_idx]

                # ========== Smoothing: Handle zero frequency ==========
                # If this bin never appeared in training for this (class, pixel)
                if likelihood == 0:
                    # Use minimum non-zero count as pseudo-count
                    # This avoids log(0) = -infinity
                    likelihood = np.min(image[j][k][np.nonzero(image[j][k])])

                # Add log likelihood: log P(Xi=bin_idx | Y=j)
                # Mathematical: log( count(Y=j, Xi=bin) / count(Y=j, pixel=k) )
                prob[j] += np.log(likelihood / image_sum[j][k])

        # ========== Normalization (optional, doesn't affect argmax) ==========
        summation = sum(prob)
        prob /= summation

        # Print result and count errors
        from io import StringIO
        capture = StringIO()
        sys.stdout = capture
        error += PrintResult(prob, answer)
        sys.stdout = sys.__stdout__
        output.append(capture.getvalue())

    # ========== Print Imagination and Error Rate ==========
    from io import StringIO
    capture = StringIO()
    sys.stdout = capture
    DrawImagination(image, image_row, image_col, 0)
    print(f'Error rate: {error / test_size}')
    sys.stdout = sys.__stdout__
    output.append(capture.getvalue())

    # Save or print output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
        print(f"[INFO] Results written to {output_file}")
    else:
        print(''.join(output))


def ContinuousMode(output_file=None):
    """
    Train and evaluate Naive Bayes classifier using Continuous mode.

    Continuous Mode assumes each pixel value follows a Gaussian distribution
    for each class. Parameters (mean, variance) are estimated using MLE.

    Args:
        output_file (str, optional): If provided, save results to this file

    Returns:
        None

    Mathematical Model:
        Prior: P(Y=j) = count(Y=j) / N
        Likelihood: P(Xi|Y=j) = N(μ_jk, σ²_jk)
            where N is Gaussian distribution

        Gaussian PDF:
            P(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

        Log Gaussian PDF:
            log P(x) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]

        Posterior:
            log P(Y=j|X) = log P(Y=j) + Σ_k log P(Xk|Y=j)

    MLE Estimation:
        μ = (1/N) Σ x_i          (sample mean)
        σ² = (1/N) Σ x_i² - μ²   (sample variance)

    Data Structures:
        mean[class][pixel]: Mean μ_jk of pixel k for class j
        var[class][pixel]: Variance σ²_jk of pixel k for class j
        prior[class]: Number of training samples for each class
    """
    print("[INFO] Starting Continuous Naive Bayes training and evaluation...")

    # Load data
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    # ========== Initialize data structures ==========
    image_size = image_row * image_col  # 784 pixels

    # prior[class]: count of each class
    prior = np.zeros((10), dtype=np.int32)

    # mean[class][pixel]: will store μ_jk
    mean = np.zeros((10, image_size), dtype=float)

    # mean_square[class][pixel]: will store E[X²] for variance calculation
    mean_square = np.zeros((10, image_size), dtype=float)

    # var[class][pixel]: will store σ²_jk
    var = np.zeros((10, image_size), dtype=float)

    output = []  # Store output for file writing

    # ========== Training Phase: Accumulate statistics ==========
    print("[INFO] Training: Accumulating statistics for Gaussian parameters...")
    for i in range(train_size):
        if i % 1000 == 0:
            print(f"[TRAIN] Processing image {i}/{train_size}")

        # Read label
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)

        # Update prior count
        prior[label] += 1

        # Read all pixels and accumulate sum and sum of squares
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)

            # Accumulate Σ x_i for mean calculation
            mean[label][j] += grayscale

            # Accumulate Σ x_i² for variance calculation
            # Mathematical: Var(X) = E[X²] - (E[X])²
            mean_square[label][j] += grayscale ** 2

    # ========== Calculate Gaussian parameters (μ, σ²) ==========
    print("[INFO] Computing Gaussian parameters (mean and variance)...")
    for i in range(10):  # For each class
        for j in range(image_size):  # For each pixel

            # Calculate mean: μ = (Σ x_i) / N
            # Mathematical: μ_ij = mean[i][j] / prior[i]
            mean[i][j] /= prior[i]

            # Calculate E[X²]: mean_square = (Σ x_i²) / N
            mean_square[i][j] /= prior[i]

            # Calculate variance: σ² = E[X²] - μ²
            # Mathematical: σ²_ij = E[X²] - (E[X])²
            var[i][j] = mean_square[i][j] - mean[i][j] ** 2

            # ========== Handle zero variance ==========
            # If all training samples have the same value, variance = 0
            # Set to large value (1000) to avoid division by zero
            if var[i][j] == 0:
                var[i][j] = 1000

    # ========== Testing Phase ==========
    print("[INFO] Starting testing...")
    error = 0  # Count prediction errors

    for i in range(test_size):
        if i % 100 == 0:
            print(f"[TEST] Evaluating test image {i}/{test_size}")

        # Read ground truth label
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)

        # Initialize log-posterior probabilities
        prob = np.zeros((10), dtype=float)

        # Read test image
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)

        # ========== Calculate Posterior for each class ==========
        for j in range(10):  # For each class j

            # Add log prior: log P(Y=j)
            prob[j] += np.log(prior[j] / train_size)

            # Add log likelihood for each pixel
            for k in range(image_size):  # For each pixel k

                # Get Gaussian parameters for this (class, pixel)
                μ = mean[j][k]     # Mean
                σ² = var[j][k]     # Variance
                x = test_image[k]  # Observed pixel value

                # ========== Calculate Log Gaussian PDF ==========
                # Gaussian PDF: P(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
                # Log PDF: log P(x) = log(1/√(2πσ²)) + log(exp(...))
                #                   = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
                #                   = -0.5 * [log(2πσ²) + (x-μ)²/σ²]

                likelihood = -0.5 * (
                    np.log(2 * math.pi * σ²) +  # log(2πσ²)
                    ((x - μ) ** 2) / σ²          # (x-μ)²/σ²
                )

                # Add to total log probability
                prob[j] += likelihood

        # ========== Normalization ==========
        summation = sum(prob)
        prob /= summation

        # Print result and count errors
        from io import StringIO
        capture = StringIO()
        sys.stdout = capture
        error += PrintResult(prob, answer)
        sys.stdout = sys.__stdout__
        output.append(capture.getvalue())

    # ========== Print Imagination and Error Rate ==========
    from io import StringIO
    capture = StringIO()
    sys.stdout = capture
    DrawImagination(mean, image_row, image_col, 1)
    print(f'Error rate: {error / test_size}')
    sys.stdout = sys.__stdout__
    output.append(capture.getvalue())

    # Save or print output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
        print(f"[INFO] Results written to {output_file}")
    else:
        print(''.join(output))


# ========== Entry Point ==========
if __name__ == '__main__':
    """
    Main entry point for the program.

    Usage:
        Training mode:
            python 2-1_naive_bayes_classifier.py --train
            Then input 0 (Discrete) or 1 (Continuous)

        Display mode:
            python 2-1_naive_bayes_classifier.py
            Then input 0 (Discrete) or 1 (Continuous)
    """
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        # Training mode: Train and save results
        mode = input("Generate Discrete(0) or Continuous(1): ")
        if mode == '0':
            DiscreteMode(output_file='Discrete')
        elif mode == '1':
            ContinuousMode(output_file='Continuous')
        else:
            print("[ERROR] Invalid input. Must be 0 or 1.")
    else:
        # Display mode: Load and display saved results
        mode = input('Discrete(0) or continuous(1): ')
        if mode == '0':
            if not os.path.exists('Discrete'):
                print("Please Train by command: python 2-1_naive_bayes_classifier.py --train")
            else:
                with open('Discrete', 'r', encoding='utf-8') as f:
                    print(f.read())
        elif mode == '1':
            if not os.path.exists('Continuous'):
                print("Please Train by command: python 2-1_naive_bayes_classifier.py --train")
            else:
                with open('Continuous', 'r', encoding='utf-8') as f:
                    print(f.read())
        else:
            print('[ERROR] Invalid input. Must be 0 or 1.')
