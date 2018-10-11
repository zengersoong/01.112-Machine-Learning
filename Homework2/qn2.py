import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

n_colors = 32
pic = 'sutd.png'
img = mpimg.imread(pic)
img = img[:,:,:3]

w, h, d = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))

def recreate_image(palette, labels, w, h):
    d = palette.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = palette[labels[label_idx]]
            label_idx += 1
    return image

    plt.figure(1)

# Part (a)
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans_palette = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
kmeans_labels = kmeans_palette.predict(image_array)


# Part (b)
random_palette = shuffle(image_array, random_state=0)[:n_colors]
random_labels = pairwise_distances_argmin(random_palette,
                                        image_array,
                                        axis=0)

plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (16.8 million colors)')
plt.imshow(img)
plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])

plt.axis('off')
plt.title('Compressed image (K-Means)')
plt.imshow(recreate_image(kmeans_palette.cluster_centers_, kmeans_labels, w, h))
plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])

plt.axis('off')
plt.title('Compressed image (Random)')
plt.imshow(recreate_image(random_palette, random_labels, w, h))
plt.show()
