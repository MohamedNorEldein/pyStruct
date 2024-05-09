import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Compute the Fourier coefficients for a given function
def fourier_coefficients(x, y, n_terms):
    """
    Compute Fourier coefficients up to n_terms.
    x: The x-axis data points.
    y: The y-axis function values corresponding to x.
    n_terms: The number of terms for which to compute coefficients.
    """
    T = x[-1] - x[0]  # Period of the function
    
    an = []
    bn = []
    w = []
    for n in range(0, n_terms + 1):
        w.append(2 * np.pi * n  / T)
        if n == 0:
            an_term = (1 / T) * np.trapz(y, x)  # Constant term
        else:
            an_term = (2 / T) * np.trapz(y * np.cos(2 * np.pi * n * x / T), x)
        bn_term = (2 / T) * np.trapz(y * np.sin(2 * np.pi * n * x / T), x)
        an.append(an_term)
        bn.append(bn_term)

    return an, bn, w


# Reconstruct the function using Fourier series
def fourier_series(x, an, bn):
    """
    Reconstruct the function from Fourier coefficients.
    """
    n_terms = len(an)
    f_approx = np.full_like(x, 0)  # Start with the constant term

    T = x[-1] - x[0]  # Period of the function

    for n in range(n_terms):
        f_approx += an[n] * np.cos(2 * np.pi * n * x / T) + bn[n] * np.sin(2 * np.pi * n * x / T)

    return f_approx

# Define the function to approximate
def original_function(x):
    return np.sin(x) + 0.5 * np.sin(2 * x)


def FourierMain():
    # Generate sample points
    x = np.linspace(0, 2 * np.pi, 1000)  # Points in one period
    y = original_function(x)

    # Compute Fourier coefficients
    a0, an, bn = fourier_coefficients(x, y, 5)  # Use 5 terms in the Fourier series

    # Reconstruct the function with the Fourier series
    y_approx = fourier_series(x, a0, an, bn)

    # Plot the original function and the Fourier series approximation
    plt.figure()
    plt.plot(x, y, label="Original Function")
    plt.plot(x, y_approx, label="Fourier Series Approximation", linestyle='--')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Fourier Series Approximation")
    plt.legend()
    plt.show()

import csv
def read_csv_file(file_path):
    """
    Read a CSV file with a header row and two columns of numerical data.
    
    Args:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - data (list of tuples): The numerical data from the CSV file.
    """
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip the header row
        next(csv_reader)
        for row in csv_reader:
            # Convert the numerical data to floats and append to the data list
            data.append((float(row[0]), float(row[1])))
    return data