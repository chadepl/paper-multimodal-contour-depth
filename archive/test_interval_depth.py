
from time import time
import numpy as np

def relu(x):
    return x if x >= 0 else 0

def interval_counting_1(numbers, index):
    count = 0
    target_num = numbers[index]
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j:
                if a <= target_num and b >= target_num:
                    count += 1
    return count


def interval_counting_1_soft(numbers, index):
    count = 0
    target_num = numbers[index]
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j:
                count += (1 - relu(a - target_num)/a) * (1 - relu(target_num - b)/target_num)
    return count


def interval_counting_2(numbers, index):
    n_below = 0
    n_above = 0
    target_num = numbers[index]
    for i, n in enumerate(numbers):
        if i != index:
            if target_num >= n:
                n_below += 1
            if target_num <= n:
                n_above += 1
    return n_above * n_below + len(numbers) - 1

def interval_counting_2_soft(numbers, index):
    n_below = 0
    n_above = 0
    target_num = numbers[index]
    for i, n in enumerate(numbers):
        if i != index:
            n_below += (1 - relu(n - target_num)/n)
            n_above += (1 - relu(target_num - n)/target_num)
    return n_above * n_below + len(numbers) - 1

# numbers = np.random.rand(10000)
numbers = np.arange(1, 100)
np.random.shuffle(numbers)

print(numbers[:10])
t_tick = time()
res1 = interval_counting_1_soft(numbers, 0)
print(f"Slow outputs {res1} and takes {time() - t_tick} seconds")
t_tick = time()
res2 = interval_counting_2_soft(numbers, 0)
print(f"Fast outputs {res2} and takes {time() - t_tick} seconds")

vals1 = []
for n, _ in enumerate(numbers):
    vals1.append(interval_counting_1_soft(numbers, n))

vals2 = []
for n, _ in enumerate(numbers):
    vals2.append(interval_counting_2_soft(numbers, n))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2)

# axs[0].hist(vals1)
# axs[1].hist(vals2)
# axs[0].hist(vals1)
# axs[1].hist(vals2)
plt.show()