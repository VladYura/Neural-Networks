lst = [-9, -5, -2, -1, 1, 4, 9, 11]


# Перебор всех пар
# Время работы: O(n**2)
# Память: O(1)
def enumeration(num, k=3):
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            if num[i] + num[j] == k:
                return [num[i], num[j]]
    return []


# HashSet
# Время работы: O(n)
# Память: O(n)
def hash_set(num, k=3):
    hash = []
    for i in range(len(num)):
        numberToFind = k - num[i]
        if numberToFind in hash:
            return [numberToFind, num[i]]
        hash.append(num[i])
    return []


# Бинарный поиск
# Время работы: O(n*log(n))
# Память: O(1)
def binary_find(num, k=3):
    num = num.sort()
    for i in range(len(num)):
        numberToFind = k - num[i]
        l, r = i + 1, len(num) - 1
        while l <= r:
            mid = l + int((r - l) / 2)
            if num[mid] == numberToFind:
                return [lst[i], lst[mid]]
            if numberToFind < lst[mid]:
                r = mid - 1
            else:
                l = mid + 1
    return []


# Два указателя
# Время работы: O(n)
# Память: O(1)
def two_pointers(num, k=3):
    num = num.sort()
    l, r = 0, len(num) - 1
    while l < r:
        if num[l] + num[r] == k:
            return [num[l], num[r]]
        elif num[l] + num[r] > k:
            r -= 1
        elif num[l] + num[r] < k:
            l += 1
    return []


print(two_pointers(lst))
