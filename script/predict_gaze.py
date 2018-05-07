import caffe
import numpy as np
import scipy.io as sio
from scipy.misc import imresize
from skimage.io import imread
from math import exp

def predict_gaze(img, e, net):
    network = caffe.Net('/home/pieter/projects/engagement\
            -l2tor/data/model/deploy_demo.prototxt', \
            '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel',\
            caffe.TEST)
    #s = RandStream(mstring('mt19937ar'), mstring('Seed'), sum(10000 * clock))
    #RandStream.setGlobalStream(s)
    np.random.seed(1)
    np.random.rand()
    #filelist = cell(1, 3)
    filelist = np.zeros((1,3))
    filelist[:, 0] = img
    #filelist(mslice[:], 1).lvalue = mcellarray([img])
    alpha = 0.3
    w_x = np.floor(alpha * img.shape[1])
    w_y = np.floor(alpha * img.shape[0])

    if (w_x % 2 == 0):
        w_x = w_x + 1
    if (w_y % 2 == 0):
        w_y = w_y + 1
    im_face = np.ones((w_y, w_x, 3))
    im_face[:,:,1] = 123*np.ones((w_y,w_x))
    im_face[:,:,2] = 117*np.ones((w_y,w_x))
    im_face[:,:,3] = 104*np.ones((w_y,w_x))

    center = np.floor([e[0]*img.shape[1] e[1]*img.shape[0]])

    d_x = np.floor((w_x-1)/2)
    d_y = np.floor((w_y-1)/2)

    bottom_x = center[0] - d_x
    delta_b_x = 1
    if (bottom_x < 1):
        delta_b_x = 2 - bottom_x
        bottom_x = 1
    top_x = center[0] + d_x
    delta_t_x = w_x

    if (top_x > img.shape[1]):
        delta_t_x = w_x - (top_x - img.shape[1])
        top_x = img.shape[1]
    bottom_y = center[1] - d_y
    delta_b_y = 1
    if (bottom_y < 1):
        delta_b_y = 2 - bottom_y
        bottom_y = 1
    top_y = center[1] + d_y
    delta_t_y = w_y
    if (top_y > img.shape[0]):
        delta_t_y = w_y - (top_y - img.shape[0])
        top_y = img.shape[0]

    im_face[delta_b_y:delta_t_y,delta_b_x:delta_t_x,:] = img[bottom_y:top_y,bottom_x:top_x,:]
    filelist[:,1] = im_face

    im_face[delta_b_y:delta_t_y,delta_b_x:delta_t_x,:] = img[bottom_y:top_y,bottom_x:top_x,:]
    filelist[:,2] = im_face

    f = np.zeros((1, 1, 169))
    z = np.zeros((13,13))
    x = np.floor(e[0] * 13) + 1
    y = np.floor(e[1] * 13) + 1
    z[x,y] = 1
    f[1,1]= z[:]
    filelist[:3,] = f

    use_gpu = 1
    device_id = 0

    transform_data = [1, 1, 0]

    caffe.set_mode_gpu()
    caffe.set_device(0)

    places_mean_resize = sio.loadmat('places_mean_resize.mat')
    imagenet_mean_resize = sio.loadmat('imagenet_mean_resize.mat')

    image_mean_cell = [places_mean_resize, imagenet_mean_resize, places_mean_resize]
    input_dim_all = [1,3, 227, 227, 1, 3, 227, 227, 1, 169, 1, 1]

    ims = np.zeros((filelis[1], 1))

    for j in range(3):
        filelist_i = filelist[0, j]
        filelist_i = filelist_i[0]
        input_dim = input_dim_all[1+(j-1)*4:j*4]
        b = np.zeros((1,1))
        img_size = [input_dim[3], input_dim[4], input_dim[2]];
        image_mean = image_mean_cell[j]

        if transform_data[j]:
            tmp = image_mean
            image_mean = np.array([image_mean], dtype=np.uint8)
            image_mean = imresize(image_mean, [img_size[0] img_size[1]])
            #done till here
            img = np.zeros((image_mean.shape[0], image_mean.shape[1], 3))

            if filelist_i.isalpha():
                img = imread(filelist_i)
            else:
                img = imread(filelist_i)
            if img.shape[3] != 1:
                img = img[:,:,:, 1]
                print(img)
                img = imresize(img, [image_mean.shape[0], image_mean.shape[1]])
            if img.shape[3] == 1:
                img = np.array(img, img, img)
            img = img.take([2,1,0], axis=2)-image_mean;
            b[0] = np.transpose(np.expand_dims(img, axis=3), (1, 0, 2))
        else:
            b[0] filelist_i
        b = np.concatenate((4, b))
        ims[j] = b

    f_val = network.forward(ims)

    fc_0_0 = f_val[0]
    fc_1_0 = f_val[1]
    fc_m1_0 = f_val[2]
    fc_0_1 = f_val[3]
    fc_0_m1 = f_val[4]

    hm = np.zeros((15, 15))
    hm = np.zeros((15, 15))

    f_0_0 = np.reshape(f_0_0(0,:), (5,5), order="F")
    
    #done till here
    f_0_0 = exp(alpha*f_0_0)/exp(alpha*f_0_0(:)).sum()

    f_1_0= reshape(fc_1_0(1,:),[5 5]);
    f_1_0 = exp(alpha*f_1_0)/sum(exp(alpha*f_1_0(:)));
    f_m1_0 = reshape(fc_m1_0(1,:),[5 5]);
    f_m1_0 = exp(alpha*f_m1_0)/sum(exp(alpha*f_m1_0(:)));
    f_0_m1 =reshape(fc_0_m1(1,:),[5 5]);
    f_0_m1 = exp(alpha*f_0_m1)/sum(exp(alpha*f_0_m1(:)));
    f_0_1 = reshape(fc_0_1(1,:),[5 5]);
    f_0_1 = exp(alpha*f_0_1)/sum(exp(alpha*f_0_1(:)));

    f_cell = {f_0_0,f_1_0,f_m1_0,f_0_m1,f_0_1};
    v_x = [0 1 -1 0 0];
    v_y = [0 0 0 -1 1];


    for k in range(5):
        delta_x = v_x(k)
        delta_y = v_y(k)
        f = f_cell(k)
        for x in mslice[1:5]:
            for y in mslice[1:5]:
                i_x = 1 + 3 * (x - 1) - delta_x
                i_x = max(i_x, 1); print i_x
                if (x == 1):
                    i_x = 1
                i_y = 1 + 3 * (y - 1) - delta_y
                i_y = max(i_y, 1); print i_y
                if (y == 1):
                    i_y = 1
                f_x = 3 * x - delta_x
                f_x = min(15, f_x); print f_x
                if (x == 5):
                    f_x = 15
                f_y = 3 * y - delta_y
                f_y = min(15, f_y); print f_y
                if (y == 5):
                    f_y = 15
                hm(mslice[i_x:f_x], mslice[i_y:f_y]).lvalue = hm(mslice[i_x:f_x], mslice[i_y:f_y]) + f(x, y)
                count_hm(mslice[i_x:f_x], mslice[i_y:f_y]).lvalue = count_hm(mslice[i_x:f_x], mslice[i_y:f_y]) + 1
    hm_base = hm /eldiv/ count_hm
    hm_results = imresize(hm_base.cT, mcat([size(img, 1), size(img, 2)]), mstring('bicubic'))
    [maxval, idx] = max(hm_results(mslice[:]))
    [row, col] = ind2sub(size(hm_results), idx)
    y_predict = row / size(hm_results, 1)
    x_predict = col / size(hm_results, 2)

if __name__=="__main__":
    caffe.set_device(0)
    caffe.set_mode_gpu()
