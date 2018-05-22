import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.misc import imresize
from matplotlib.patches import Circle
import sys
sys.path.append('/home/pieter/projects/caffe/python')
import caffe

def loadModel():
    model_def = '/home/pieter/projects/engagement-l2tor/data/model/deploy_demo.prototxt'
    model_weights = '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel'
    network = caffe.Net(model_def, model_weights, caffe.TEST)
    return network

def prepImages(img, e):
    """
    Output images of prepImages are exactly the same as the matlab ones

    Keyword Arguments:
    img --  image with subject for gaze calculation
    e --  head location (relative) [x, y]
    """
    input_shape = [227, 227]
    alpha = 0.3
    img = imread(img)
    img_resize = None
    #height, width
    #crop of face (input 2)
    wy = int(alpha * img.shape[0])
    wx = int(alpha * img.shape[1])
    center = [int(e[0]*img.shape[1]), int(e[1]*img.shape[0])]
    y1 = int(center[1]-.5*wy) - 1
    y2 = int(center[1]+.5*wy) - 1
    x1 = int(center[0]-.5*wx) - 1
    x2 = int(center[0]+.5*wx) - 1
    #make crop of face from image
    im_face = img[y1:y2, x1:x2, :]

    #subtract mean from images
    places_mean = sio.loadmat('data/model/places_mean_resize.mat')
    imagenet_mean = sio.loadmat('data/model/imagenet_mean_resize.mat')
    places_mean = places_mean['image_mean']
    imagenet_mean = imagenet_mean['image_mean']

    #resize image and subtract mean
    img_resize = imresize(img, input_shape, interp='bicubic')
    img_resize = img_resize.astype('float32')
    img_resize = img_resize[:,:,[2,1,0]] - places_mean
    img_resize = np.rot90(np.fliplr(img_resize))

    #resize eye image
    eye_image = imresize(im_face, input_shape, interp='bicubic')
    eye_image = eye_image.astype('float32')
    eye_image_resize = eye_image[:,:,[2,1,0]] - imagenet_mean
    eye_image_resize = np.rot90(np.fliplr(eye_image_resize))
    #get everything in the right input format for the network
    img_resize, eye_image_resize = fit_shape_of_inputs(img_resize, eye_image_resize)
    z = eyeGrid(img, [x1, x2, y1, y2])
    z = z.astype('float32')
    return img, img_resize, eye_image_resize, z

def fit_shape_of_inputs(img_resize, eye_image_resize):
    """Fits the input for the forward pass."""
    input_image_resize = img_resize.reshape([img_resize.shape[0], \
                                               img_resize.shape[1], \
                                               img_resize.shape[2], 1])
    input_image_resize = input_image_resize.transpose(3, 2, 0, 1)

    eye_image_resize = eye_image_resize.reshape([eye_image_resize.shape[0], \
                                                eye_image_resize.shape[1], \
                                                eye_image_resize.shape[2], 1])
    eye_image_resize = eye_image_resize.transpose(3, 2, 0, 1)
    return input_image_resize, eye_image_resize

def eyeGrid(img, headlocs):
    """Calculates the relative location of the eye.

    Keyword Arguments:
    img -- original image
    headlocs -- relative head location
    """
    w = img.shape[1]
    h = img.shape[0]
    x1_scaled = headlocs[0] / w
    x2_scaled = headlocs[1] / w
    y1_scaled = headlocs[2] / h
    y2_scaled = headlocs[3] / h
    center_x = (x1_scaled + x2_scaled) * 0.5
    center_y = (y1_scaled + y2_scaled) * 0.5
    eye_grid_x = np.floor(center_x * 12).astype('int')
    eye_grid_y = np.floor(center_y * 12).astype('int')
    eyes_grid = np.zeros([13, 13]).astype('int')
    eyes_grid[eye_grid_y, eye_grid_x] = 1
    eyes_grid_flat = eyes_grid.flatten()
    eyes_grid_flat = eyes_grid_flat.reshape(1, len(eyes_grid_flat), 1, 1)
    return eyes_grid_flat

def predictGaze(network, image, head_image, head_loc):
    """Loads data in network and does a forward pass."""
    network.blobs['data'].data[...] = image
    network.blobs['face'].data[...] = head_image
    network.blobs['eyes_grid'].data[...] = head_loc
    f_val = network.forward()
    return f_val

def postProcessing(f_val):
    """Combines the 5 outputs into one heatmap and calculates the gaze location

    Keyword arguments:
    f_val -- output of the Caffe model
    """
    fc_0_0 = f_val['fc_0_0'].T
    fc_0_1 = f_val['fc_0_1'].T
    fc_m1_0 = f_val['fc_m1_0'].T
    fc_0_1 = f_val['fc_0_1'].T
    fc_0_m1 = f_val['fc_0_m1'].T
    f_0_0 = np.reshape(fc_0_0, (5,5))
    f_1_0 = np.reshape(fc_0_1, (5,5))
    f_m1_0 = np.reshape(fc_m1_0, (5,5))
    f_0_1 = np.reshape(fc_0_1, (5,5))
    f_0_m1 = np.reshape(fc_0_m1, (5,5))
    gaze_grid_list = [alpha_exponentiate(f_0_0), \
                          alpha_exponentiate(f_1_0), \
                          alpha_exponentiate(f_m1_0), \
                          alpha_exponentiate(f_0_1), \
                          alpha_exponentiate(f_0_m1)]
    shifted_x = [0, 1, -1, 0, 0]
    shifted_y = [0, 0, 0, -1, 1]
    count_map = np.ones([15, 15])
    average_map = np.zeros([15, 15])
    for delta_x, delta_y, gaze_grids in zip(shifted_x, shifted_y, gaze_grid_list):
        for x in range(0, 5):
            for y in range(0, 5):
                ix = shifted_mapping(x, delta_x, True)
                iy = shifted_mapping(y, delta_y, True)
                fx = shifted_mapping(x, delta_x, False)
                fy = shifted_mapping(y, delta_y, False)
                average_map[ix:fx+1, iy:fy+1] += gaze_grids[x, y]
                count_map[ix:fx+1, iy:fy+1] += 1
    average_map = average_map / count_map
    final_map = imresize(average_map, (227,227), interp='bicubic')
    idx = np.argmax(final_map.flatten())
    [rows, cols] = ind2sub2((227, 227), idx)
    y_predict = rows/227
    x_predict = cols/227
    return final_map, [x_predict, y_predict]

def alpha_exponentiate(x, alpha=0.3):
    return np.exp(alpha * x) / np.sum(np.exp(alpha*x.flatten()))

def ind2sub2(array_shape, ind):
    """Python implementation of the equivalent matlab method"""
    rows = (ind / array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return [rows, cols]

def shifted_mapping(x, delta_x, is_topleft_corner):
    if is_topleft_corner:
        if x == 0:
            return 0
        ix = 0 + 3 * x - delta_x
        return max(ix, 0)
    else:
        if x == 4:
            return 14
        ix = 3 * (x + 1) - 1 - delta_x
    return min(14, ix)

def getGaze(e, image):
    """Calculate the gaze direction in an imageself.

    Keyword arguments:
    e -- list with x,y location of head
    image -- original image
    """
    network = loadModel()
    image, image_resize, head_image, head_loc = prepImages(image, e)
    f_val = predictGaze(network, image_resize, head_image, head_loc)
    final_map, predictions = postProcessing(f_val)
    x = predictions[0] * np.shape(image)[0]
    y = predictions[1] * np.shape(image)[1]
    x = int(x)
    y = int(y)
    return [x,y]

if __name__=="__main__":
    #this main method is for testing purposes
    #predictions = getGaze([0.60, 0.2679], 'script/test.jpg')
    predictions = getGaze([0.54, 0.28], 'script/5.jpg')
    image = imread('script/5.jpg')
    fig, ax = fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(image)
    ax.add_patch(Circle((predictions[0], predictions[1]),10))
    plt.show()
