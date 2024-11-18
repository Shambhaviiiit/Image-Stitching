import os
import numpy as np
import cv2
import time
from scipy.sparse.linalg import lsqr
from scipy import linalg
from src.siftMatch import sift_match
from src.multiSample_APAP import multi_sample_apap
from LineMatching.twoLineMatch import two_line_match
from src.calcHomoPointLine import calc_homo_point_line
from src.generateUV import generate_uv
from src.energyLineV import energy_line_v
from src.energyLineU import energy_line_u
from src.energyAlign import energy_align
from src.energyLineAlign import energy_line_align
from src.linesDetect import lines_detect
from src.energyLineSegment import energy_line_segment
from src.texture_mapping.meshmap_warp2homo import meshmap_warp2homo
from src.imageBlending import image_blending

# Placeholder function definitions assumed to be implemented elsewhere
# from your_custom_library import add_need_paths, sift_match, multi_sample_apap, two_line_match, calc_homo_point_line
# from your_custom_library import generate_uv, energy_line_v, energy_line_u, energy_align, energy_line_align
# from your_custom_library import lines_detect, energy_line_segment, meshmap_warp2homo, image_blending, vec2mat

# Add paths and toolbox
# add_need_paths()

# Parameters for energy minimization (mesh deformation)
parameters = {
    'grid_size': 40,
    'line_align': 5,
    'perspective': 50,
    'projective': 5,
    'saliency': 5,
    'line_threshold': 50
}
# Images to stitch
tar = 2
ref_n = 1
pathname = 'src/Images/0/'
# print(os.getcwd())
outpath = os.path.join(pathname, 'results/')
# print(outpath)
# if not os.path.exists(outpath):
#     os.makedirs(outpath)

imgs_format = '*.jpg'
# entries = os.listdir(os.getcwd())

# # Print each entry
# for entry in entries:
#     print(entry)

dir_folder = [f for f in os.listdir(pathname) if f.endswith('.jpg')]
path1 = os.path.join(pathname, dir_folder[tar - 1])
path2 = os.path.join(pathname, dir_folder[ref_n - 1])

# Read images
print('> Reading images...', end='')
start_time = time.time()
img1 = cv2.imread(path1).astype(np.float64) / 255
img2 = cv2.imread(path2).astype(np.float64) / 255
img1 = np.uint8(img1 * 255) 
img2 = np.uint8(img2 * 255) 
img1 = img1.astype('uint8')
img2 = img2.astype('uint8')

print(f'done ({time.time() - start_time:.2f}s)')

# Resolution/grid-size for mapping function
C1 = int(np.ceil(img1.shape[0] / parameters['grid_size']))
C2 = int(np.ceil(img1.shape[1] / parameters['grid_size']))

# Detect and match SIFT features
pts1, pts2 = sift_match(img1, img2)
matches_1, matches_2 = multi_sample_apap(pts1, pts2)
print("AFTER MATCHES")
print(matches_1)

# Detect and match line segments
line_match1, line_match2 = two_line_match(img1, img2, matches_1, matches_2, parameters)

# Single-perspective warp (SPW) and blending
print('  Our SPW warp and blending...', end='')
start_time = time.time()
h, _, T1, T2 = calc_homo_point_line(matches_1, matches_2, line_match1, line_match2)
pts_line_H = np.linalg.inv(T2) @ (h @ T1)

# Generating mesh grid
X, Y = np.meshgrid(np.linspace(1, img1.shape[1], C2 + 1), np.linspace(1, img1.shape[0], C1 + 1))
Mv = np.column_stack((X.ravel(), Y.ravel()))

init_H = pts_line_H / pts_line_H[-1, -1]
theta = np.arctan2(-init_H[1, 2], -init_H[0, 2])

# Generate u-v sample points and normal vector
print('> generating u-v sample points and u-v term...', end='')
start_time = time.time()
lines_vs, lines_us, lines_ue = generate_uv(img1, img2, init_H, theta, C1, C2)
nor_vec_v = np.array([init_H[1, 1] * init_H[1, 2] - init_H[0, 1] * init_H[0, 2],
                      init_H[0, 2] * init_H[0, 0] - init_H[0, 0] * init_H[1, 2]])
nor_vec_v /= np.linalg.norm(nor_vec_v)
sparse_v = energy_line_v(img1, C1, C2, lines_vs, nor_vec_v)
sparse_us, sparse_ue = energy_line_u(img1, C1, C2, lines_us, lines_ue, init_H)
print(f'done ({time.time() - start_time:.2f}s)')

# Alignment energy term with scale operator
print('> generating scale-alignment term...', end='')
start_time = time.time()
sparse_align, psMatch = energy_align(img1, C1, C2, matches_1, matches_2)
sparse_line_align, cMatch = energy_line_align(img1, C1, C2, line_match1, line_match2)
print(f'done ({time.time() - start_time:.2f}s)')

# Line-preserving term with line segments
print('> detect line segments...', end='')
start_time = time.time()
sa_lines, sl_lines = lines_detect(path1, img1, C1, C2)
sparse_line = energy_line_segment(img1, sa_lines, sl_lines, init_H, C1, C2)
print(f'done ({time.time() - start_time:.2f}s)')

# Construct matrix A and b
zero_len = sparse_us.shape[0] + sparse_v.shape[0] + sparse_ue.shape[0] + sparse_line.shape[0]
warp_hv = init_H @ np.vstack((Mv.T, np.ones(Mv.shape[0])))
warp_hv /= warp_hv[2, :]
init_V = warp_hv[:2, :].flatten()

para_l = parameters['line_align']
para_ps = parameters['perspective']
para_pj = parameters['projective']
para_s = parameters['saliency']

Matrix_A = np.vstack([sparse_align, np.sqrt(para_l) * sparse_line_align, np.sqrt(para_ps) * sparse_us,
                      np.sqrt(para_ps) * sparse_v, np.sqrt(para_pj) * sparse_ue, np.sqrt(para_s) * sparse_line])
m_x = np.concatenate([psMatch, np.sqrt(para_l) * cMatch, np.zeros(zero_len)])

# Solve energy minimization
print('> Use LSQR method...', end='')
start_time = time.time()
V_star, flag, _, iter_count = lsqr(Matrix_A, m_x, atol=1e-8, btol=1e-8, iter_lim=5000, x0=init_V)[:4]
print(f'done ({time.time() - start_time:.2f}s)')

optimized_V = V_star.reshape(-1, 2)

# Mesh deformation using bilinear interpolation
print('> mesh deformation...', end='')
start_time = time.time()
wX = optimized_V[:, 0].reshape(C1 + 1, C2 + 1)
wY = optimized_V[:, 1].reshape(C1 + 1, C2 + 1)
warped_img1 = meshmap_warp2homo(img1, X, Y, wX, wY)
print(f'done ({time.time() - start_time:.2f}s)')

# Blend the warped images and save the result
off = np.ceil([1 - min(1, *optimized_V[:, 0]), 1 - min(1, *optimized_V[:, 1])])
cw = int(max(max(optimized_V[:, 0]), img2.shape[1]) + off[0] - 1)
ch = int(max(max(optimized_V[:, 1]), img2.shape[0]) + off[1] - 1)

img1Homo = np.zeros((ch, cw, 3))
img2Homo = np.zeros((ch, cw, 3))

img1Homo[int(min(optimized_V[:, 1]) + off[1] - 1):int(min(optimized_V[:, 1]) + off[1] - 1 + warped_img1.shape[0]),
         int(min(optimized_V[:, 0]) + off[0] - 1):int(min(optimized_V[:, 0]) + off[0] - 1 + warped_img1.shape[1]), :] = warped_img1
img2Homo[int(off[1]):int(off[1] + img2.shape[0]), int(off[0]):int(off[0] + img2.shape[1]), :] = img2

linear_out = image_blending(img1Homo, img2Homo, 'linear')
pngout = f'linear-{para_l}-{para_ps}-{para_pj}-{para_s}.jpg'
cv2.imwrite(os.path.join(outpath, pngout), (linear_out * 255).astype(np.uint8))

print('done.')
