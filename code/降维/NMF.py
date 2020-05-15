import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn import decomposition
from PIL import Image

Tupian = Image.open('test.jpg')

Lupian = Tupian.convert('L')
Lupian.save("gray.jpg")

data = []
image = img.imread('gray.jpg')
print(image.shape)
data.append(image)

n_row, n_col = 2,3
n_components = 6
image_shape = (24,30)

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) 
    plt.suptitle(title, size=16)
 
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
 
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)
    plt.savefig(str(name))

estimators = [('Eigenfaces - PCA using randomized SVD',decomposition.PCA(n_components=1,whiten=True)),\
    ('Non-negative components - NMF',decomposition.NMF(n_components=1,init='nndsvda',tol=5e-3))]

for name,estimator in estimators:
    print("Extracting the top %d %s..." % (n_components,name))
    print(image.shape)
    estimator.fit(image)
    components_ = estimator.components_
    print(components_)
    print(len(components_[0]))
    plot_gallery(name, components_[:n_components])

plt.show()