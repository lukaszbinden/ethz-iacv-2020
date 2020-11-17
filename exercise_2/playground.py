import numpy as np

x = np.arange(10)
print(x)
print(np.roll(x, 2))
print(np.roll(x, -2))


x2 = np.reshape(x, (2,5))
print(x2)
#print(np.roll(x2, 1))
print("---")
print(np.roll(x2, 1, axis=0))
print(np.roll(x2, -1, axis=0))
print("---")
print(np.roll(x2, 1, axis=1))
print(np.roll(x2, -1, axis=1))
print("----------------------------------------------------")



m_height = 8
m_width = 10
mask_halfwidth = 2
import numpy as np
ncc = np.zeros((m_height - 2*mask_halfwidth, m_width - 2*mask_halfwidth, m_width - 2*mask_halfwidth))
ncc[:,:,0] = 0.5
ncc[:,:,3] = 1
ncc[:,:,4] = 2

new_width = m_width - 2 * mask_halfwidth
new_height = m_height - 2 * mask_halfwidth

# for y in range(0, new_height):
#     # for each pixel in the left image...
#     for x_l in range(1, new_width):  # start from 1 since x_r < x_l
#         # ... find one corresponding pixel in the right image
#         print(f"y={y}, x_l={x_l}")
#         ncc_candidates = ncc[y, x_l, 0:x_l]
#         print(f"\tncc={ncc_candidates} \t argmax={np.argmax(ncc_candidates)}")




# for y in range(0, new_height):
#     for x_l in range(1, new_width):
#             for x_r in range(0,x_l):
#                     print(f"y={y}, x_l={x_l}, x_r={x_r}")
#                     ncc_candidates = ncc[y, x_l, 0:x_r+1]
#                     print(f"\tncc={ncc_candidates} \t argmax={np.argmax(ncc_candidates)}")

points3d = np.zeros((10, 3))
points3d[2:4,:] = 3
points3d[-3:-1,:] = 8
points3d[-1,:] = 9
print(points3d)
print(">>>>>>")
points_3d_y = points3d[mask_halfwidth:-mask_halfwidth,:]
print(points_3d_y)
# d = np.zeros((points3d.shape[0] - 2*mask_halfwidth, 3))

m_height, m_width = 10, 10
new_height = m_height - 2 * mask_halfwidth
new_width = m_width - 2 * mask_halfwidth
d = np.zeros((new_height, new_width, 3))
d[mask_halfwidth + 2] = points_3d_y
print(d)
