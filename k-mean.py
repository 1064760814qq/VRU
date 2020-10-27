from skimage import io
from sklearn.cluster import  KMeans
import numpy as np

image=io.imread('tim.jpg')
# io.imshow(image)
# io.show()

rows=image.shape[0]
cols=image.shape[1]

image=image.reshape(image.shape[0]*image.shape[1],3)
kmeans=KMeans(n_clusters=128,n_init=10,max_iter=200)
kmeans.fit(image)


clusters=np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
labels=np.asarray(kmeans.labels_,dtype=np.uint8)
labels=labels.reshape(rows,cols)

print(clusters.shape)
np.save('photo_test.npy',clusters)
io.imsave('Ph_test.jpg',labels)


image=io.imread('Ph_test.jpg')
io.imshow(image)
io.show()