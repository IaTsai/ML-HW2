# Machine Learning HW2 - Naive Bayes & Online Learning

**Course**: Machine Learning (2025 Spring)
**Student**: 313553058 Ian Tsai
**Assignment**: HW2 - Naive Bayes Classifier & Bayesian Online Learning

---

## Assignment Overview

This assignment consists of three parts:

1. **HW2-1: Naive Bayes Classifier**

   - Discrete Mode: Using 32 bins to count pixel frequencies
   - Continuous Mode: Modeling with Gaussian distribution

2. **HW2-2: Online Learning**

   - Beta-Binomial Conjugation implementation
   - Online updating of coin flip probability

3. **HW2-3: Mathematical Derivation**
   - Proof of Beta-Binomial Conjugation
   - Proof of Gamma-Poisson Conjugation

---

## Project Structure

```
HW2_Dday1008/
├── 2-1_naive_bayes_classifier.py    # Naive Bayes Classifier
├── 2-2_online_learning.py           # Online Learning
├── testfile.txt                     # Test data for Online Learning
├── MNIST_dataset.zip                # MNIST dataset (compressed)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Environment Setup

### System Requirements

- Python 3.6 or higher
- Operating System: Windows / macOS / Linux

### Install Dependencies

This project only requires **NumPy**, no other special packages needed.

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Verify Environment

```bash
python --version  # Should display Python 3.x
python -c "import numpy; print(numpy.__version__)"  # Should display NumPy version
```

---

## Prepare Test Data

### Method 1: Extract MNIST Dataset (Recommended)

If you have `MNIST_dataset.zip` in the repository:

```bash
# Extract MNIST dataset
unzip MNIST_dataset.zip

# Verify files
ls -lh *.idx*-ubyte_
```

You should see 4 files:

```
t10k-images.idx3-ubyte_   (7.5 MB)  - Test images
t10k-labels.idx1-ubyte_   (9.8 KB)  - Test labels
train-images.idx3-ubyte_  (45 MB)   - Training images
train-labels.idx1-ubyte_  (59 KB)   - Training labels
```

### Method 2: Download from E3 or Official Website

If you don't have the compressed file, download from:

1. **NYCU E3 System**: Download from HW2 assignment section
2. **MNIST Official Website**: http://yann.lecun.com/exdb/mnist/
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`

After downloading, extract and rename (add underscore `_`):

```bash
gunzip train-images-idx3-ubyte.gz
mv train-images-idx3-ubyte train-images.idx3-ubyte_
# Repeat for the other 3 files
```

---

## How to Run

### HW2-1: Naive Bayes Classifier

#### Training Mode (First Time Execution)

```bash
# Run training
python 2-1_naive_bayes_classifier.py --train

# System will prompt for mode selection
Generate Discrete(0) or Continuous(1): 0  # Enter 0 or 1
```

**Description**:

- Enter `0`: Train Discrete Mode (~2-3 minutes)
- Enter `1`: Train Continuous Mode (~3-4 minutes)
- Training results will be saved to `Discrete` or `Continuous` file

#### Display Results Mode

```bash
# Load pre-trained results
python 2-1_naive_bayes_classifier.py

# System will prompt for mode selection
Discrete(0) or continuous(1): 0  # Enter 0 or 1
```

**Description**:

- Directly loads saved training results
- Displays prediction results for all test images
- Shows "Imagination" visualization for digits 0-9
- Displays error rate

#### Expected Output Example

```
Posterior (in log scale):
0: 0.11127455255545808
1: 0.11792841531242379
2: 0.1052274113969039
3: 0.10015879429196257
4: 0.09380188902719812
5: 0.09744539128015761
6: 0.1145761939658308
7: 0.07418582789605557
8: 0.09949702276138589
9: 0.08590450151262384
Prediction: 7, Ans: 7

...

Imagination of numbers in Bayesian classifier:
0:
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
...

Error rate: 0.1583
```

---

### HW2-2: Online Learning

```bash
# Run Online Learning
python 2-2_online_learning.py testfile.txt

# System will prompt for Beta Prior parameters
input a: 0   # Enter a parameter
input b: 0   # Enter b parameter
```

#### Test Cases

**Case 1: Uniform Prior (No prior knowledge)**

```bash
python 2-2_online_learning.py testfile.txt
input a: 0
input b: 0
```

**Case 2: Informative Prior (With prior knowledge)**

```bash
python 2-2_online_learning.py testfile.txt
input a: 10
input b: 1
```

#### Expected Output Example

```
case 1: 0101010101001011010101
Likelihood: 0.16818809509277344
Beta prior:     a = 0  b = 0
Beta posterior: a = 11  b = 11

case 2: 0110101
Likelihood: 0.29375515303997485
Beta prior:     a = 11  b = 11
Beta posterior: a = 15  b = 14

...
```

---

## Performance Metrics

### Naive Bayes Classifier

| Mode       | Training Time | Error Rate |
| ---------- | ------------- | ---------- |
| Discrete   | ~2-3 minutes  | ~15-16%    |
| Continuous | ~3-4 minutes  | ~15-17%    |

**Test Environment**: 60,000 training images, 10,000 test images

### Online Learning

- **Execution Time**: < 1 second
- **Test Cases**: 11 lines of binary sequences

---

## Mathematical Derivation

For complete mathematical proofs, please refer to:

1. **Markdown Version**: `HW2_Report.md`

   - Contains detailed code explanations
   - Includes both conjugate prior proofs

2. **LaTeX Version**: `main.tex`
   - Suitable for uploading to Overleaf
   - Or use `HW2_LaTeX_Final_313553058.zip`

### Compile on Overleaf

1. Upload `HW2_LaTeX_Final_313553058.zip` to Overleaf
2. Select "New Project" → "Upload Project"
3. Compiler setting: pdfLaTeX
4. Compile to generate PDF

---

## Troubleshooting

### Q1: MNIST Data Files Not Found

**Error Message**:

```
FileNotFoundError: [Errno 2] No such file or directory: './train-images.idx3-ubyte_'
```

**Solution**:

```bash
# Check if files exist
ls *.idx*-ubyte_

# If not, extract MNIST_dataset.zip
unzip MNIST_dataset.zip
```

### Q2: NumPy Not Installed

**Error Message**:

```
ModuleNotFoundError: No module named 'numpy'
```

**Solution**:

```bash
pip install numpy
```

### Q3: Python Version Too Old

**Error Message**:

```
SyntaxError: invalid syntax
```

**Solution**:
Ensure Python version ≥ 3.6

```bash
python --version
# Or try
python3 --version
```

### Q4: Training Result Files Already Exist

To retrain, delete old result files:

```bash
rm Discrete Continuous
python 2-1_naive_bayes_classifier.py --train
```

### Q5: Out of Memory

Continuous Mode may use more memory (~1-2 GB). If memory is insufficient:

- Close other applications
- Or only run Discrete Mode

---

## Code Documentation

### 2-1_naive_bayes_classifier.py

**Main Functions**:

- `LoadImage()` / `LoadLabel()`: Load MNIST binary data
- `DiscreteMode()`: Discrete Naive Bayes implementation
- `ContinuousMode()`: Continuous Naive Bayes implementation
- `DrawImagination()`: Visualize digit features

**Core Algorithm**:

```python
# Discrete Mode
prob[j] += log(prior[j] / train_size)
prob[j] += log(likelihood / image_sum[j][k])

# Continuous Mode
likelihood = -0.5 * (log(2π×σ²) + (x-μ)²/σ²)
prob[j] += likelihood
```

### 2-2_online_learning.py

**Main Functions**:

- `Factorial()`: Factorial calculation
- `C()`: Combination calculation
- Beta-Binomial Conjugation update

**Core Algorithm**:

```python
# Posterior update
a' = a + m  # m = number of successes
b' = b + n  # n = number of failures
```

---

## References

1. **MNIST Dataset**

   - http://yann.lecun.com/exdb/mnist/

2. **Naive Bayes**

   - Pattern Recognition and Machine Learning (Bishop)
   - Chapter 4: Linear Models for Classification

3. **Conjugate Priors**
   - Pattern Recognition and Machine Learning (Bishop)
   - Chapter 2: Probability Distributions

---

## License

This project is for NYCU Machine Learning course assignment use only.

**Note**: Please do not plagiarize or directly use for other course assignments.

---

**Last Updated**: 2025/01/24
