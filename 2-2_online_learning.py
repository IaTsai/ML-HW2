"""
Beta-Binomial Bayesian Online Learning

This program demonstrates online learning using Beta-Binomial conjugation
to estimate the probability parameter p of a coin flip from observed data.

Mathematical Foundation:
    Beta-Binomial Conjugation:
        Prior: p ~ Beta(a, b)
        Likelihood: X ~ Binomial(N, p), where X = number of successes
        Posterior: p ~ Beta(a + m, b + n)
            where m = number of successes (1s)
                  n = number of failures (0s)

    Key Property:
        The posterior distribution is also a Beta distribution!
        This allows simple parameter updates: a' = a + m, b' = b + n

    Binomial Likelihood (MLE):
        P(X=m | N, p) = C(N, m) * p^m * (1-p)^(N-m)
        where p = m/N (maximum likelihood estimate)
              C(N, m) = N! / (m! * (N-m)!)  (binomial coefficient)

Author: 313553058 Ian Tsai
Course: Machine Learning HW2 (2025 Spring)
"""

import sys


def Factorial(n):
    """
    Calculate factorial of n recursively.

    Args:
        n (int): Non-negative integer

    Returns:
        int: n! = n * (n-1) * (n-2) * ... * 2 * 1

    Mathematical Formula:
        n! = n * (n-1)!
        0! = 1
        1! = 1

    Examples:
        Factorial(0) = 1
        Factorial(1) = 1
        Factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
    """
    # Base case: 0! = 1, 1! = 1
    # Recursive case: n! = n * (n-1)!
    return n * Factorial(n - 1) if n > 1 else 1


def C(N, m):
    """
    Calculate binomial coefficient C(N, m) = "N choose m".

    Args:
        N (int): Total number of items
        m (int): Number of items to choose

    Returns:
        float: Binomial coefficient C(N, m) = N! / (m! * (N-m)!)

    Mathematical Formula:
        C(N, m) = N! / (m! * (N-m)!)
                = N * (N-1) * ... * (N-m+1) / m!

    Optimization:
        Uses the property C(N, m) = C(N, N-m) to minimize computation.
        If N-m < m, we swap to compute the smaller factorial.

    Examples:
        C(5, 2) = 5! / (2! * 3!) = 10
        C(10, 3) = 10! / (3! * 7!) = 120
        C(N, 0) = 1 (choose nothing)
        C(N, N) = 1 (choose everything)
    """
    # Optimization: C(N, m) = C(N, N-m)
    # Choose the smaller value to reduce computation
    if N - m < m:
        m = N - m

    # Calculate: N * (N-1) * ... * (N-m+1)
    res = 1
    for i in range(m):
        res *= N  # Multiply by N, then N-1, then N-2, ...
        N -= 1

    # Divide by m! to get final result
    # Mathematical: C(N, m) = [N * (N-1) * ... * (N-m+1)] / m!
    res /= Factorial(m)

    return res


if __name__ == '__main__':
    """
    Main entry point for Beta-Binomial online learning.

    Usage:
        python 2-2_online_learning.py <testfile>

        Then input:
            - a: Shape parameter of Beta prior
            - b: Shape parameter of Beta prior

    Test Cases:
        1. Uniform Prior (no prior knowledge):
           a = 0, b = 0

        2. Informative Prior (prior belief: p is high):
           a = 10, b = 1

    Mathematical Process:
        For each line of binary data (e.g., "0101010101"):
            1. Count successes m (number of 1s)
            2. Count failures n (number of 0s)
            3. Calculate MLE: p_MLE = m / N
            4. Calculate Binomial likelihood: P(D|p) = C(N,m) * p^m * (1-p)^n
            5. Update Beta parameters:
               a' = a + m
               b' = b + n
    """

    # ========== Check command-line arguments ==========
    if len(sys.argv) < 2:
        print('Usage: python 2-2_online_learning.py <testfile>')
    else:
        # ========== Load test data ==========
        # Read all lines from the test file
        # Each line contains a binary string (e.g., "0101010101")
        with open(sys.argv[1], 'r', encoding='utf-8') as file:
            testcase = file.readlines()

        # ========== Get Beta prior parameters ==========
        # Beta(a, b) represents our prior belief about parameter p
        # - a = 0, b = 0: Uniform prior (no prior knowledge)
        # - a = 10, b = 1: Strong prior belief that p is high (biased towards 1)
        # - a = 1, b = 10: Strong prior belief that p is low (biased towards 0)
        a = int(input('input a: '))
        b = int(input('input b: '))

        # ========== Online Learning: Process each test case ==========
        for i in range(len(testcase)):
            # Remove whitespace from line
            line = testcase[i].strip()

            # ========== Count successes and failures ==========
            N = len(line)  # Total number of trials (length of binary string)

            # Count number of successes (1s)
            # Mathematical: m = Σ I(x_i = 1)
            m = 0
            for char in line:
                m += 1 if char == '1' else 0

            # Number of failures (0s)
            # Mathematical: n = N - m
            n = N - m

            # ========== Calculate MLE (Maximum Likelihood Estimate) ==========
            # MLE for Binomial distribution: p_MLE = m / N
            # This is the empirical probability of seeing a 1
            P = m / N

            # ========== Calculate Binomial Likelihood ==========
            # Binomial Likelihood: P(D | p) = C(N, m) * p^m * (1-p)^n
            # where:
            #   - C(N, m): Binomial coefficient "N choose m"
            #   - p: Probability parameter (we use MLE estimate p = m/N)
            #   - m: Number of successes
            #   - N-m: Number of failures
            #
            # Mathematical Formula:
            #   P(X = m | N, p) = [N! / (m! * (N-m)!)] * p^m * (1-p)^(N-m)
            likelihood = C(N, m) * (P ** m) * ((1 - P) ** (N - m))

            # ========== Print Results ==========
            print(f'case {i + 1}: {line}')
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior:     a = {a}  b = {b}')

            # ========== Beta-Binomial Conjugate Update ==========
            # Update Beta parameters using conjugacy property:
            #   Prior: Beta(a, b)
            #   Data: m successes, n failures
            #   Posterior: Beta(a + m, b + n)
            #
            # Mathematical Proof (from HW2 report):
            #   P(p|D) ∝ P(D|p) * P(p)
            #          ∝ [p^m * (1-p)^n] * [p^(a-1) * (1-p)^(b-1)]
            #          = p^(a+m-1) * (1-p)^(b+n-1)
            #          = Beta(a+m, b+n)

            # Update parameter a: add number of successes
            a += m

            # Update parameter b: add number of failures
            b += n

            print(f'Beta posterior: a = {a}  b = {b}\n')

            # The updated posterior becomes the prior for the next iteration!
            # This is the "online" aspect of online learning.
