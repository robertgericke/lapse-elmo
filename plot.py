import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (11.7,8.3)
plt.rcParams["figure.dpi"] = 600
data = np.loadtxt('access.txt')

plt.plot(data)
plt.savefig('plot.png')
plt.yscale('log')
plt.savefig('plot_log.png')

plt.clf()

sort = np.flip(np.sort(data))
plt.plot(sort)
plt.savefig('sort.png')
plt.yscale('log')
plt.savefig('sort_log.png')

plt.clf()

points = data[0:793470]
sampled = data[1586981:2380449]
plt.plot(points)
plt.plot(sampled)
plt.savefig('word_distribution.png')
plt.yscale('log')
plt.savefig('word_distribution_log.png')