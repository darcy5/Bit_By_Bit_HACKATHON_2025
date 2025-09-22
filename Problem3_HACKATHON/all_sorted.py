import numpy as np

sorted_keys = np.load('sorted_keys.npy')

keys_list = sorted_keys.tolist()


with open("all_sorted_keys.txt", "w") as f:
    for k in keys_list:
        f.write(str(k) + "\n")

print(sorted_keys.tolist())
print("All sorted keys saved to all_sorted_keys.txt")

