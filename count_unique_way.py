import numpy as np
# Время работы: O(2**(n + m))
def paths(n, m):
    if n < 1 or m < 1:
        return 0
    if n == 1 and m == 1:
        return 1
    return paths(n - 1, m) + paths(n, m - 1)


# Время работы: O(n*m)
def paths2(n, m):
    return helper(n, m, np.zeros((n + 1, m + 1)))


def helper(n, m, arr):
    if n < 1 or m < 1:
        return 0
    if n == 1 and m == 1:
        return 1
    if arr[n][m] != 0:
        return arr[n][m]
    arr[n][m] = helper(n - 1, m, arr) + helper(n, m - 1, arr)
    return arr[n][m]


print(paths(4, 5))
print(paths2(3, 3))
