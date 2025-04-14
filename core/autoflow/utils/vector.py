def cosine_distance(v1, v2):
    import numpy as np

    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
