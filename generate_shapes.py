import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os


# 2d rotation matrix
def R2d(theta):
    return numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])


# create centered polygons vertices (triangle, square, pentagon, hexagon)
yhat = numpy.array([0, 1])
ploygons = [numpy.array([R2d(i * 2 * numpy.pi / dim) @ yhat for i in range(dim)]) for dim in [3, 4, 5, 6]]

images = []
dpi = 64
N = 200000
xmin, xmax, ymin, ymax = 0, 100, 0, 100
# read ground truth values for the shape parameters
# all parameters have uniform distribution
# with the following range:
# shape_idx ~ [0,1,2,3], theta ~ [0, 2pi],
# s ~ [20,40], r ~ [0,1], g~ [0,1], b~ [0,1]
gts = numpy.load('gts.npy')

with tqdm(total=N) as pbar:
    for gt in gts:
        shape_idx, theta, s, r, g, b = gt
        plot_poly = s * R2d(theta) @ ploygons[int(shape_idx)].T + numpy.array([50, 50])[:, None]
        dpi = 64
        fig = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.fill(*plot_poly, c=(r, g, b))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('off')
        canvas.draw()
        image = numpy.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(64, 64, 3).astype(numpy.int32)
        images.append(image)
        pbar.update()

os.makedirs('data/shapes', exist_ok=True)
numpy.savez('data/shapes/shapes.npz', images=numpy.array(images), gts=gts)

# print sample of the generate polygons
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), dpi=64)
i = 0
for row in ax:
    for col in row:
        col.imshow(images[i])
        col.axis('off')
        i += 1
plt.show()
