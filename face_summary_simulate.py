import numpy as np
import time

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


face_feats = np.load("./examples/faces_feats.npy")
cosine_distance_matrix = np.load("./examples/cosine_dist.npy")
euclidean_l2_distance_matrix = np.load("./examples/euclidean_l2_dist.npy")


t1 = time.time()
cos_dis = cosine_distances(face_feats, face_feats)
t2 = time.time()

print('Time cost = {} ms'.format(1000.0 * (t2 - t1)))


t3 = time.time()


norm_vector = np.linalg.norm(face_feats, axis=-1)
norm_vector = np.expand_dims(norm_vector, axis=-1)
faces_norm = face_feats / norm_vector
eud_dis = euclidean_distances(faces_norm, faces_norm)
t4 = time.time()
print('Time cost = {} ms'.format(1000.0 * (t4 - t3)))

print(cos_dis.shape)
print(eud_dis.shape)

error_cos = cosine_distance_matrix - cos_dis
error_eud = euclidean_l2_distance_matrix - eud_dis

print(np.sum(np.abs(error_cos)))
print(np.sum(np.abs(error_eud)))
