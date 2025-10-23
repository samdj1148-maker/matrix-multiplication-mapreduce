#!/usr/bin/env python3
"""
Matrix Multiplication using Simulated MapReduce Logic

This program demonstrates matrix multiplication using MapReduce concepts:
- Map phase: Generate intermediate (key, value) pairs for each element
- Reduce phase: Aggregate pairs by key to compute final matrix elements
"""

import sys
from collections import defaultdict


def read_matrices_from_file(filename):
    """
    Read two matrices from input file.
    Expected format:
    - First matrix as rows of space-separated numbers
    - Empty line separator
    - Second matrix as rows of space-separated numbers
    """
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Find the separator between matrices (empty line handled by strip)
        # Split matrices by finding where dimensions change or by counting
        matrix_a = []
        matrix_b = []
        
        # Read first matrix
        idx = 0
        first_row_len = None
        while idx < len(lines):
            row = [float(x) for x in lines[idx].split()]
            if first_row_len is None:
                first_row_len = len(row)
            # If we encounter a row with different length, it might be matrix B
            if len(row) != first_row_len and matrix_a:
                break
            matrix_a.append(row)
            idx += 1
        
        # Read second matrix
        while idx < len(lines):
            row = [float(x) for x in lines[idx].split()]
            matrix_b.append(row)
            idx += 1
        
        return matrix_a, matrix_b
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def map_phase(matrix_a, matrix_b):
    """
    MAP PHASE:
    For matrix multiplication C = A × B where:
    - A is m×n matrix
    - B is n×p matrix
    - C is m×p matrix
    
    Map emits: key=(i,j), value=(matrix_id, k, element_value)
    For each A[i][k], emit ((i,j), ('A', k, A[i][k])) for all j in range(p)
    For each B[k][j], emit ((i,j), ('B', k, B[k][j])) for all i in range(m)
    """
    print("\n=== MAP PHASE ===")
    
    m = len(matrix_a)  # rows in A
    n = len(matrix_a[0])  # cols in A = rows in B
    p = len(matrix_b[0])  # cols in B
    
    # Validate dimensions
    if len(matrix_b) != n:
        raise ValueError(f"Matrix dimensions incompatible: A is {m}×{n}, B is {len(matrix_b)}×{p}")
    
    mapped_values = defaultdict(list)
    
    # Map from matrix A
    print(f"\nMapping from Matrix A ({m}×{n}):")
    for i in range(m):
        for k in range(n):
            for j in range(p):
                key = (i, j)
                value = ('A', k, matrix_a[i][k])
                mapped_values[key].append(value)
                if i < 2 and j < 2 and k < 2:  # Print sample mappings
                    print(f"  A[{i}][{k}]={matrix_a[i][k]} -> key=({i},{j}), value={value}")
    
    # Map from matrix B
    print(f"\nMapping from Matrix B ({n}×{p}):")
    for k in range(n):
        for j in range(p):
            for i in range(m):
                key = (i, j)
                value = ('B', k, matrix_b[k][j])
                mapped_values[key].append(value)
                if i < 2 and j < 2 and k < 2:  # Print sample mappings
                    print(f"  B[{k}][{j}]={matrix_b[k][j]} -> key=({i},{j}), value={value}")
    
    print(f"\nTotal mapped keys: {len(mapped_values)}")
    return mapped_values, (m, p)


def reduce_phase(mapped_values, dimensions):
    """
    REDUCE PHASE:
    For each key (i,j), collect all mapped values and compute:
    C[i][j] = Σ(A[i][k] × B[k][j]) for all k
    
    Groups values by key, matches A and B values with same k, multiplies and sums.
    """
    print("\n=== REDUCE PHASE ===")
    
    m, p = dimensions
    result_matrix = [[0.0 for _ in range(p)] for _ in range(m)]
    
    for key, values in sorted(mapped_values.items()):
        i, j = key
        
        # Separate A and B values
        a_values = {k: val for (matrix, k, val) in values if matrix == 'A'}
        b_values = {k: val for (matrix, k, val) in values if matrix == 'B'}
        
        # Compute dot product: sum of A[i][k] * B[k][j] for all k
        dot_product = 0.0
        for k in a_values.keys():
            if k in b_values:
                dot_product += a_values[k] * b_values[k]
        
        result_matrix[i][j] = dot_product
        
        # Print sample reductions
        if i < 2 and j < 2:
            print(f"  Reducing key ({i},{j}):")
            for k in sorted(a_values.keys()):
                if k in b_values:
                    print(f"    A[{i}][{k}]={a_values[k]} × B[{k}][{j}]={b_values[k]} = {a_values[k] * b_values[k]}")
            print(f"    Result C[{i}][{j}] = {dot_product}")
    
    return result_matrix


def write_matrix_to_file(matrix, filename):
    """
    Write the result matrix to output file.
    """
    try:
        with open(filename, 'w') as f:
            for row in matrix:
                f.write(' '.join(str(elem) for elem in row) + '\n')
        print(f"\nResult written to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)


def print_matrix(matrix, name):
    """
    Pretty print a matrix.
    """
    print(f"\n{name}:")
    for row in matrix:
        print("  [", end="")
        print(", ".join(f"{elem:6.1f}" for elem in row), end="")
        print(" ]")


def main():
    """
    Main execution function:
    1. Read matrices from input.txt
    2. Perform MapReduce-style multiplication
    3. Write result to output.txt
    """
    print("=" * 60)
    print("Matrix Multiplication with Simulated MapReduce")
    print("=" * 60)
    
    # Read input matrices
    input_file = 'input.txt'
    output_file = 'output.txt'
    
    print(f"\nReading matrices from {input_file}...")
    matrix_a, matrix_b = read_matrices_from_file(input_file)
    
    # Display input matrices
    print_matrix(matrix_a, "Matrix A")
    print_matrix(matrix_b, "Matrix B")
    
    # Perform MapReduce multiplication
    try:
        # Map phase
        mapped_values, dimensions = map_phase(matrix_a, matrix_b)
        
        # Reduce phase
        result_matrix = reduce_phase(mapped_values, dimensions)
        
        # Display result
        print_matrix(result_matrix, "\nResult Matrix C = A × B")
        
        # Write to output file
        write_matrix_to_file(result_matrix, output_file)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"  Matrix A dimensions: {len(matrix_a)}×{len(matrix_a[0])}")
        print(f"  Matrix B dimensions: {len(matrix_b)}×{len(matrix_b[0])}")
        print(f"  Result C dimensions: {len(result_matrix)}×{len(result_matrix[0])}")
        print(f"  Output saved to: {output_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during computation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
