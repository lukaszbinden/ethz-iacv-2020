from scipy import spatial
import numpy as np


pi = np.pi
bins = np.linspace(-pi, pi, num=9)
print(bins)

K = 10
points = np.zeros((36000, 128))
res = np.random.randint(0, points.shape[0], K)
print(points[res].shape)

points_1 = np.array([[1,2],
                     [3,4],
                     [1,6]])
points_2 = np.array([[2,2], [5,5], [3,1]])
res = spatial.distance.cdist(points_1, points_2, metric='euclidean')

print(f"euclid: \n{res.shape}\n{res}")

am = np.argmin(res, axis=1)
print(f"argmin = {am}")

ind = [0, 1, 2, 3, 4, 5, 5, 1, 0, 0]
res = np.array(ind)[np.where(np.array(ind) == 0)]
print(res)

mean = np.mean(points_1, axis=0)
print(f"mean: {mean}")

labels = 0 + np.zeros((500, 1))
labels = np.concatenate((labels, 1 + np.zeros((500, 1))))
print(labels)
print(np.sum(labels[0:500]))
print(np.sum(labels[500:1000]))
nn_indices = np.array([334, 319,  51, 694,  49, 311,  95,  921,  60, 731])
print("labels of test images:")
print(labels[nn_indices])

print("-j-------------------------")
labels = 0 + np.zeros((4, 1))
labels = np.concatenate((labels, 1 + np.zeros((4, 1))))
print(labels)
print(np.sum(labels[0:4]))
print(np.sum(labels[4:8]))
nn_indices = np.array([4, 7,  1, 6,  4, 3,  5,  5,  0, 7])
print("labels of test images:")
print(labels[nn_indices])
