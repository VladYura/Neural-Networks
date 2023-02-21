import numpy as np

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


print(smooth_curve(np.array([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
])))
