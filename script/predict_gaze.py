import caffe
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage.io import imread
from skimage.transform import resize
from math import exp
from matplotlib.patches import Circle

def predict_gaze(img, e):
    network = caffe.Net('/home/pieter/projects/engagement-l2tor/data/model/deploy_demo.prototxt', \
            '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel',\
            caffe.TEST)
    np.random.seed(1)
    np.random.rand()
    filelist = []
    filelist.append(img)
    alpha = 0.3
    w_x = np.floor(alpha * img.shape[1])
    w_y = np.floor(alpha * img.shape[0])
    w_x = int(w_x)
    w_y = int(w_y)

    if (w_x % 2 == 0):
        w_x = w_x + 1
    if (w_y % 2 == 0):
        w_y = w_y + 1

    im_face = np.ones((w_y, w_x, 3))
    im_face[:,:,0] = 123*np.ones((w_y,w_x))
    im_face[:,:,1] = 117*np.ones((w_y,w_x))
    im_face[:,:,2] = 104*np.ones((w_y,w_x))

    center = np.floor([e[0]*img.shape[1],e[1]*img.shape[0]])

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


    im_face[delta_b_y:delta_t_y , delta_b_x:delta_t_x,:] = img[int(bottom_y):int(top_y),int(bottom_x):int(top_x),:]
    filelist.append(im_face)

    f = np.zeros((1, 1, 169))
    z = np.zeros((13,13))
    x = np.floor(e[0] * 13) + 1
    y = np.floor(e[1] * 13) + 1
    z[int(x),int(y)] = 1
    f[0,0,:]= z.flatten(1)
    filelist.append(f)

    use_gpu = 1
    device_id = 0

    transform_data = [1, 1, 0]
    places_mean_resize = sio.loadmat('data/model/places_mean_resize.mat')
    imagenet_mean_resize = sio.loadmat('data/model/imagenet_mean_resize.mat')

    image_mean_cell = [places_mean_resize, imagenet_mean_resize, places_mean_resize]
    input_dim_all = [1,3, 227, 227, 1, 3, 227, 227, 1, 169, 1, 1]

    ims = [None] * len(filelist)

    for j in range(0,3):
        filelist_i = filelist[j]
        #filelist_i = filelist_i[0]
        input_dim = input_dim_all[0+((j+1)-1)*4:(j+1)*4]
        b = []
        img_size = [input_dim[2], input_dim[3], input_dim[1]]
        image_mean = image_mean_cell[j]['image_mean']
        if transform_data[j]:
            tmp = image_mean
            image_mean = imresize(image_mean, (img_size[0], img_size[1]))
            img = np.zeros((image_mean.shape[0], image_mean.shape[1], 3))
            #tested till here
            img = filelist_i
            #if img.shape[3] != 1:
            #    img = img[:,:,:, 1]
            img = imresize(img, [image_mean.shape[0], image_mean.shape[1]])
            #if img.shape[2] == 1:
            #    img = np.array(img, img, img)
            img = img - image_mean
            b.append(np.transpose(img, (1,0,2)))
        else:
            b.append(filelist_i)
        b = np.hstack(b)
        ims[j] = b
    #validated untill here


    network.blobs['data'] = ims[0]
    network.blobs['face'] = ims[1]
    network.blobs['eyes_grid'] = ims[2]

    f_val = network.forward()

    print(f_val.keys())

    fc_0_0 = f_val['fc_0_0']
    fc_1_0 = f_val['fc_1_0']
    fc_m1_0 = f_val['fc_m1_0']
    fc_0_1 = f_val['fc_0_1']
    fc_0_m1 = f_val['fc_0_m1']

    hm = np.zeros((15, 15))
    count_hm = np.zeros((15, 15))

    #this line is correct
    f_0_0 = np.reshape(fc_0_0[0,], (5,5), order="F")
    f_0_0 = np.exp(alpha * f_0_0)/np.exp(alpha*np.sum(f_0_0))

    f_1_0 = np.reshape(fc_1_0[0,], (5,5), order="F")
    f_1_0 = np.exp(alpha*f_1_0)/np.sum(np.exp(alpha*np.sum(f_1_0)))

    f_m1_0 = np.reshape(fc_m1_0[0,], (5,5), order="F")
    f_m1_0 = np.exp(alpha*f_m1_0)/np.sum(np.exp(alpha*np.sum(f_m1_0)))

    f_0_m1 = np.reshape(fc_0_m1[0,], (5,5), order="F")
    f_0_m1 = np.exp(alpha*f_0_m1) / np.sum(np.exp(alpha*np.sum(f_0_m1)))

    f_0_1 = np.reshape(fc_0_1[0,], (5,5), order="F")
    f_0_1 = np.exp(alpha*f_0_1) / np.sum(np.exp(alpha*np.sum(f_0_1)))

    f_cell = [f_0_0,f_1_0,f_m1_0,f_0_m1,f_0_1]
    v_x = [0, 1, -1, 0, 0]
    v_y = [0, 0, 0, -1, 1]

    for k in range(5):
        delta_x = v_x[k]
        delta_y = v_y[k]
        f = f_cell[k]
        for x in range(5):
            for y in range(5):
                i_x = 1 + 3 * (x - 1) - delta_x
                i_x = max(i_x, 1)
                #print(i_x)
                if (x == 1):
                    i_x = 1
                i_y = 1 + 3 * (y - 1) - delta_y
                i_y = max(i_y, 1)
                #print(i_y)
                if (y == 1):
                    i_y = 1
                f_x = 3 * x - delta_x
                f_x = min(15, f_x)
                #print(f_x)
                if (x == 5):
                    f_x = 15
                f_y = 3 * y - delta_y
                f_y = min(15, f_y)
                #print(f_y)
                if (y == 5):
                    f_y = 15
                hm[i_x:f_x, i_y:f_y] = hm[i_x:f_x, i_y:f_y] + f[x,y]
                count_hm[i_x:f_x, i_y:f_y] = count_hm[i_x:f_x, i_y:f_y] + 1

    hm_base = hm / count_hm
    hm_base = np.array(hm_base, dtype=np.uint8)
    hm_results = imresize(hm_base, (img.shape[0], img.shape[1]), interp='bicubic')

    hmr = hm_results.flatten()
    maxval, idx = hmr.max(0), hmr.argmax(0)
    #[maxval, idx] = np.max(hm_results.flatten())
    print(idx)
    rows_cols = ind2sub2(np.shape(hm_results), idx)

    y_predict = rows_cols[0] / np.shape(hm_results[0])
    x_predict = rows_cols[1] / np.shape(hm_results[1])
    return x_predict, y_predict, hm_results, network

def ind2sub2(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return [rows, cols]

def visualize(x_predict, y_predict, img):
    g = np.floor([x_predict, y_predict]*[img.shape[1], img.shape[0]])
    e = np.floor(e * [img.shape[1], img.shape[0]])
    line = [e[0], e[1], g[0], g[1]]
    print(line)
    #im = insertShape(im,'line',line,'Color','red','LineWidth',8);

if __name__=="__main__":
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    e = [0.6, 0.2679]
    img = imread("script/test.jpg")
    x_predict, y_predict, hm_results, network = predict_gaze(img, e)
    # Show the image
    print(np.shape(img))
    print(x_predict, y_predict)
    im = plt.imread('script/test.jpg')
    fig, ax = fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    #multiply with corresponding axes to go to absolute coordinates
    circ = Circle((x_predict*1400,y_predict*2100),50)
    ax.add_patch(circ)
    ax.add_patch(Circle((e[0]*2100, e[1]*1400), 50))
    plt.show()
