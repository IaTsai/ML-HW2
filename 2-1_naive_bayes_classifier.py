import sys, math, binascii
import numpy as np
import os

# Load image file from binary MNIST dataset
def LoadImage(file):
    print(f"[INFO] Loading image file: {file}")
    image_file = open(f'./{file}', 'rb')  # Changed from './data/{file}' to './{file}'
    image_file.read(4)  # magic number
    return image_file

# Load label file from binary MNIST dataset
def LoadLabel(file):
    print(f"[INFO] Loading label file: {file}")
    label_file = open(f'./{file}', 'rb')  # Changed from './data/{file}' to './{file}'
    label_file.read(4)  # magic number
    return label_file

# Print log-scale posterior and return prediction error status
def PrintResult(prob, answer):
    print('Posterior (in log scale):')
    for i in range(prob.shape[0]):
        print(f'{i}: {prob[i]}')
    pred = np.argmin(prob)
    print(f'Prediction: {pred}, Ans: {answer}\n')
    return 0 if answer == pred else 1

# Print visual imagination of digits based on training model
def DrawImagination(image, image_row, image_col, mode):
    print('Imagination of numbers in Bayesian classifier\n')
    if mode == 0:
        for i in range(10):
            print(f'{i}:')
            for j in range(image_row):
                for k in range(image_col):
                    white = sum(image[i][j * image_row + k][:16])
                    black = sum(image[i][j * image_row + k][16:])
                    print(f'{1 if black > white else 0} ', end='')
                print()
            print()
    elif mode == 1:
        for i in range(10):
            print(f'{i}:')
            for j in range(image_row):
                for k in range(image_col):
                    print(f'{1 if image[i][j * image_row + k] > 128 else 0} ', end='')
                print()
            print()

# Load training image and label data
def LoadTrainingData():
    print("[INFO] Loading training data...")
    train_image_file = LoadImage('train-images.idx3-ubyte_')
    train_size = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_row = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    train_label_file = LoadLabel('train-labels.idx1-ubyte_')
    train_label_file.read(4)
    print(f"[INFO] Training size: {train_size}, Image size: {image_row}x{image_col}")
    return train_image_file, train_label_file, train_size, image_row, image_col

# Load testing image and label data
def LoadTestingData():
    print("[INFO] Loading testing data...")
    test_image_file = LoadImage('t10k-images.idx3-ubyte_')
    test_size = int(binascii.b2a_hex(test_image_file.read(4)), 16)
    test_image_file.read(4)
    test_image_file.read(4)
    test_label_file = LoadLabel('t10k-labels.idx1-ubyte_')
    test_label_file.read(4)
    print(f"[INFO] Test size: {test_size}")
    return test_image_file, test_label_file, test_size

# Train and evaluate discrete Naive Bayes classifier
def DiscreteMode(output_file=None):
    print("[INFO] Starting Discrete Naive Bayes training and evaluation...")
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col
    image = np.zeros((10, image_size, 32), dtype=np.int32)
    image_sum = np.zeros((10, image_size), dtype=np.int32)
    prior = np.zeros((10), dtype=np.int32)

    output = []
    for i in range(train_size):
        if i % 1000 == 0:
            print(f"[TRAIN] Processing image {i}/{train_size}")
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            image[label][j][grayscale // 8] += 1
            image_sum[label][j] += 1

    print("[INFO] Starting testing...")
    error = 0
    for i in range(test_size):
        if i % 100 == 0:
            print(f"[TEST] Evaluating test image {i}/{test_size}")

        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=float)
        test_image = np.zeros((image_size), dtype=np.int32)

        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)

        for j in range(10):
            prob[j] += np.log(prior[j] / train_size)
            for k in range(image_size):
                likelihood = image[j][k][test_image[k] // 8]
                if likelihood == 0:
                    likelihood = np.min(image[j][k][np.nonzero(image[j][k])])
                prob[j] += np.log(likelihood / image_sum[j][k])
        summation = sum(prob)
        prob /= summation
        from io import StringIO
        capture = StringIO()
        sys.stdout = capture
        error += PrintResult(prob, answer)
        sys.stdout = sys.__stdout__
        output.append(capture.getvalue())

    from io import StringIO
    capture = StringIO()
    sys.stdout = capture
    DrawImagination(image, image_row, image_col, 0)
    print(f'Error rate: {error / test_size}')
    sys.stdout = sys.__stdout__
    output.append(capture.getvalue())

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
        print(f"[INFO] Results written to {output_file}")
    else:
        print(''.join(output))

# Train and evaluate continuous Naive Bayes classifier
def ContinuousMode(output_file=None):
    print("[INFO] Starting Continuous Naive Bayes training and evaluation...")
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col
    prior = np.zeros((10), dtype=np.int32)
    mean = np.zeros((10, image_size), dtype=float)
    mean_square = np.zeros((10, image_size), dtype=float)
    var = np.zeros((10, image_size), dtype=float)

    output = []
    for i in range(train_size):
        if i % 1000 == 0:
            print(f"[TRAIN] Processing image {i}/{train_size}")
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            mean[label][j] += grayscale
            mean_square[label][j] += grayscale ** 2

    for i in range(10):
        for j in range(image_size):
            mean[i][j] /= prior[i]
            mean_square[i][j] /= prior[i]
            var[i][j] = mean_square[i][j] - mean[i][j] ** 2
            var[i][j] = 1000 if var[i][j] == 0 else var[i][j]

    print("[INFO] Starting testing...")
    error = 0
    for i in range(test_size):
        if i % 100 == 0:
            print(f"[TEST] Evaluating test image {i}/{test_size}")
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            prob[j] += np.log(prior[j] / train_size)
            for k in range(image_size):
                likelihood = -0.5 * (np.log(2 * math.pi * var[j][k]) + ((test_image[k] - mean[j][k]) ** 2) / var[j][k])
                prob[j] += likelihood
        summation = sum(prob)
        prob /= summation
        from io import StringIO
        capture = StringIO()
        sys.stdout = capture
        error += PrintResult(prob, answer)
        sys.stdout = sys.__stdout__
        output.append(capture.getvalue())

    from io import StringIO
    capture = StringIO()
    sys.stdout = capture
    DrawImagination(mean, image_row, image_col, 1)
    print(f'Error rate: {error / test_size}')
    sys.stdout = sys.__stdout__
    output.append(capture.getvalue())

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
        print(f"[INFO] Results written to {output_file}")
    else:
        print(''.join(output))

# ========== Entry point ==========
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        mode = input("Generate Discrete(0) or Continuous(1): ")
        if mode == '0':
            DiscreteMode(output_file='Discrete')
        elif mode == '1':
            ContinuousMode(output_file='Continuous')
        else:
            print("[ERROR] Invalid input. Must be 0 or 1.")
    else:
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