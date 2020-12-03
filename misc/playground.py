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
