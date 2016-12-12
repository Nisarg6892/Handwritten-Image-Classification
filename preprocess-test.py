import glob
import random
import numpy as np
from scipy import ndimage
from scipy import misc

image_list = glob.glob("images/*.png")
image_list.sort()
total_images = len(image_list)
# im_w = 3300
# im_h = 2548
im_w_resized = 128
im_h_resized = 128

print (total_images)


def gen_pair_tensors(img1, img2):
    pair_tensor = np.ndarray(shape=(2, im_w_resized, im_h_resized), dtype=np.float32)
    im1 = ndimage.imread(img1, flatten=True).astype(float)
    im2 = ndimage.imread(img2, flatten=True).astype(float)
    im1_resized = misc.imresize(im1, (im_w_resized, im_h_resized))
    im2_resized = misc.imresize(im2, (im_w_resized, im_h_resized))
    im1_resized = (im1_resized - 255. / 2.0) / 255.0
    im2_resized = (im2_resized - 255. / 2.0) / 255.0

    pair_tensor[0, :, :] = im1_resized
    pair_tensor[1, :, :] = im2_resized
    return pair_tensor


def gen_pair_img_data(img1, img2):
    label = np.array([1, 0])
    l1 = int(img1.split('/')[1][:4])
    l2 = int(img2.split('/')[1][:4])
    if l1 != l2:
        label = np.array([0, 1])
    return gen_pair_tensors(img1, img2), label


def get_batch(count, size):
    image_batch = np.ndarray(shape=((size - 2) * 4, 2, im_w_resized, im_h_resized), dtype=np.float32)
    label_batch = np.ndarray(shape=((size - 2) * 4, 2), dtype=np.int32)
    j = 0
    i = count * size
    end = i + size
    while i < end - 2:
        img, lab = gen_pair_img_data(image_list[i], image_list[i + 1])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        img, lab = gen_pair_img_data(image_list[i], image_list[i + 2])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        img, lab = gen_pair_img_data(image_list[i], image_list[random.randint(0, total_images - 1)])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        img, lab = gen_pair_img_data(image_list[i], image_list[random.randint(0, total_images - 1)])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        i += 1
    return image_batch, label_batch

def get_test_batch(count, size):
    image_batch = np.ndarray(shape=((size - 3) * 2, 2, im_w_resized, im_h_resized), dtype=np.float32)
    label_batch = np.ndarray(shape=((size - 3) * 2, 2), dtype=np.int32)
    j = 0
    i = count * size
    end = i + size
    while i < end - 3:
        img, lab = gen_pair_img_data(image_list[i],image_list[i + 3])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        img, lab = gen_pair_img_data(image_list[i], image_list[random.randint(0, total_images - 1)])
        image_batch[j] = img
        label_batch[j] = lab
        j += 1
        i += 1
    return image_batch, label_batch


def gen_test_data(num):
    inc = (100 - 3) * 2
    image_batch = np.ndarray(shape=(num*inc, 2, im_w_resized, im_h_resized), dtype=np.float32)
    label_batch = np.ndarray(shape=(num*inc, 2), dtype=np.int32)
    j=0
    for i in range(num):
        random_count = random.randint(0,155)
        img, lab = get_test_batch(random_count,100)
        image_batch[j:j+inc] = img
        label_batch[j:j+inc] = lab
        j += inc
    return image_batch,label_batch


def save_test_data(num):
    test_images,test_labels = gen_test_data(num)
    print test_labels.shape
    np.save('test_images.npy',test_images)
    np.save('test_labels.npy',test_labels)

save_test_data(10)

# ib,lb =  get_batch(80,20)
# print ib.shape
# print lb.shape
