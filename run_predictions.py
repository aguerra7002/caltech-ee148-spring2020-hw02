import os
import numpy as np
import json
from PIL import Image, ImageDraw

def compute_convolution(I, T, stride=None, name=""):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    padding = int((T.shape[0] - 1) / 2)
    IPad = np.zeros((n_rows + padding*2, n_cols + padding*2, 3))
    IPad[padding:-padding, padding:-padding] = I
    heatmap = np.zeros((n_rows, n_cols))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i,j,0] < 240:
                continue
            pxs = IPad[i:i + T.shape[0], j:j + T.shape[1]]
            dot_prod = np.sum(np.multiply(pxs, T))
            T_norm_x_min = max(padding - i, 0)
            T_norm_y_min = max(padding - j, 0)
            T_norm_x_max = min(T.shape[0] - padding + I.shape[0] - i, T.shape[0])
            T_norm_y_max = min(T.shape[1] - padding + I.shape[1] - j, T.shape[1])
            T_norm = np.linalg.norm(T[T_norm_x_min:T_norm_x_max, T_norm_y_min:T_norm_y_max])
            # if T_norm == 0:
            #     print("T:", T[:,:,0], T_norm_x_max, T_norm_x_min, T_norm_y_max, T_norm_y_min)
            # if np.linalg.norm(pxs) == 0:
            #     print("I:", pxs[:,:,0], i, j)
            pxs = IPad[i:i + T.shape[0], j:j + T.shape[1]]
            dot_prod = np.sum(np.multiply(pxs, T))
            cos_sim = dot_prod / (np.linalg.norm(pxs) * T_norm)
            heatmap[i, j] = cos_sim # will be between 0 and 1
    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    padding = 4
    for i in range(0, heatmap.shape[0], 20):
        for j in range(0, heatmap.shape[1], 20):
            tmp = heatmap[i:i+20, j:j+20]
            best_i, best_j = np.unravel_index(np.argmax(tmp, axis=None), (20,20))
            if tmp[best_i, best_j] > 0.85: # This is our threshold
                min_x = max(0, i + best_i - padding)
                max_x = min(heatmap.shape[0], i + best_i + padding)
                min_y = max(0, j + best_j - padding)
                max_y = min(heatmap.shape[1], j + best_j + padding)
                output.append([str(min_y), str(min_x), str(max_y), str(max_x), str(tmp[best_i,best_j])])
    fin = sorted(output, key=lambda x: x[-1])
    # Only return the first 3 if too many things pass the filter
    return fin


def detect_red_light_mf(I, name = ""):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    # For this algorithm, all of our templates are 7 * 7. This makes convolution and recovering the bounding box simple.

    # read image using PIL:
    template_path = "../data/hw02_template/"
    fin_heatmap = np.zeros((I.shape[0], I.shape[1]))
    for f in os.listdir(template_path):

        T = Image.open(os.path.join(template_path, f))

        # convert to numpy array:
        T = np.asarray(T)

        heatmap = compute_convolution(I, T, name=name)
        # Element-wise maximum for the heatmap
        fin_heatmap = np.maximum(heatmap, fin_heatmap)


    output = predict_boxes(fin_heatmap)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    print(i)
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))
    dI = ImageDraw.Draw(I)
    # convert to numpy array:
    I_np = np.asarray(I)
    preds_train[file_names_train[i]] = detect_red_light_mf(I_np, name=file_names_train[i])

    # Get the bounding boxes
    bbs = preds_train[file_names_train[i]]

    # For each bounding box
    for bb in bbs:

        # Draw the bounding box on the image
        dI.rectangle([int(e) for e in bb[:-1]])
    # Save the resulting figure.
    I.save(os.path.join("../data/hw02_drawn/", file_names_train[i]))
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)
        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
