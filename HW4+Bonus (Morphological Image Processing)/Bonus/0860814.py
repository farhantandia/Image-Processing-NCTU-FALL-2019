import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import sys
from PIL import Image

#pick one way to load the image

# img_in = Image.open('test.jpg')
img_in = Image.open(sys.argv[1])

img_in = np.array(img_in, dtype=np.float64) / 255

w, h, d = original_shape = tuple(img_in.shape)
assert d == 3
image_array = np.reshape(img_in, (w * h, d))

print("Fitting model on a small sub-sample of the data")
image_array_sample = shuffle(image_array, random_state=0)[:500]
print("Predicting color indices on the full image (affinity propagation)")
t0 = time()
af_prop = AffinityPropagation(max_iter=200, damping = 0.9,convergence_iter=50).fit(image_array_sample)
labels_af = af_prop.predict(image_array)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(img_in)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (Affinity Propagation)')
af_prop = recreate_image(af_prop.cluster_centers_, labels_af, w, h)
plt.imshow(af_prop)


#if you want the ouput of segmented image
img_out = Image.fromarray((af_prop*255).astype(np.uint8))
img_out.save('0860814_affinity_09_500.png')