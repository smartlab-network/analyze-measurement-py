import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(r"C:\labhub\Repos\smartlab-network\analyze-measurement-py\test_data\20250125a_stim0.csv", delimiter=",")
print(data.shape)
t = data[:, 0]
values = data[:, 1:]
value1 = values[:, 10]

plt.figure(figsize=(10, 5))

plt.plot(t, value1)
plt.title("Alle Messwerte")
plt.xlabel("Zeit")
plt.ylabel("Amplitude")

plt.grid()
plt.show()