import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming

targets = np.loadtxt("q_3/hamming_codes_q3_n13_k10_5.txt").astype(int)
targets = targets[2000:4000]

N = len(targets[0])  # binary code of length N
D = 7   # with minimum distance D
Q = np.max(targets) + 1    # number of symbols in alphabet
M = len(targets)  # number of unique codes in general

# construct hamming distance matrix
A = np.zeros((M, M), dtype=int)
for i in range(M):
    for j in range(i+1, M):
        A[i, j] = hamming(targets[i], targets[j]) * N
A += A.T

MIN_GROUP_SIZE = 10

def recursivly_find_legit_numbers(nums, codes=set(), groups=list()):
    if len(groups) > 0 and len(groups[-1]) < MIN_GROUP_SIZE:
        """Found enough subgroups for initial number i"""
        return set()

    unchecked = nums.copy()

    for num1 in nums:

        unchecked -= {num1}
        candidate = unchecked.copy()
        codes.add(num1)
        for num2 in unchecked:
            if A[num1, num2] < D:
                "Distance isn't sufficient, remove this number from set"
                candidate -= {num2}

        if len(candidate) > 0:
            codes = recursivly_find_legit_numbers(candidate, codes, groups)
        else:
            groups.append(codes)
            codes = set(list(codes)[:-1])

    return set()

group_of_codes = {}

for i in tqdm(range(M)):
    groups = []

    satisfying_numbers = np.where(A[i] >= D)[0]
    satisfying_numbers = satisfying_numbers[satisfying_numbers > i]
    nums = set(satisfying_numbers)

    if len(nums) == 0:
        continue
    recursivly_find_legit_numbers(nums, set(), groups)
    [subgroup.add(i) for subgroup in groups]
    group_of_codes[i] = groups

largest_group = 0
for g, group in group_of_codes.items():
    for s, subgroup in enumerate(group):
        if len(subgroup) > largest_group:
            largest_group = len(subgroup)
            ind = (g, s)

print(f"largest group for N={N} and D={D}: {largest_group}")
print("Number of unique groups:", len(group_of_codes))

f_name = f"final/final_codes_q" + str(Q) + '_n' + str(N) + '_d' + str(D) + ".txt"
f = open(f_name, "w")

for first, group in group_of_codes.items():
    for subgroup in group:
        data = ["".join(map(str, targets[num].tolist())) for num in subgroup]
        f.write(str(data)+"\n")

f.close()
