# main.py - Matrix Multiplication using MapReduce (Simulated)
def read_matrices(filename):
    with open(filename) as f:
        content = f.read().strip().split('\n\n')
        A = [list(map(int, row.split())) for row in content[0].split('\n')]
        B = [list(map(int, row.split())) for row in content[1].split('\n')]
    return A, B

def map_step(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError("Cannot multiply: matrices have wrong shapes")
    mapped = []
    for i in range(m):
        for k in range(p):
            for j in range(n):
                mapped.append(((i,k), A[i][j] * B[j][k]))
    return mapped

def reduce_step(mapped, m, p):
    from collections import defaultdict
    reduced = defaultdict(int)
    for (i, k), value in mapped:
        reduced[(i, k)] += value
    result = [[reduced[(i, k)] for k in range(p)] for i in range(m)]
    return result

def write_matrix(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
    print("Result written to", filename)
    print("Output Matrix:")
    for row in matrix:
        print(row)

if __name__ == '__main__':
    A, B = read_matrices('input.txt')
    mapped = map_step(A, B)
    result = reduce_step(mapped, len(A), len(B[0]))
    write_matrix(result, 'output.txt')
