import numpy as np
import cv2

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def mask_and_downsample(image, prediction):
    #select only relevant pixels

    index = np.where(prediction == 0)
    
    #mask the non-relevant part of the image
    masked_image = image.astype('float')
    masked_image[index] = -1.

    # Resize image
    h, w, _ = masked_image.shape
    w_new = int(100 * w / max(w, h) )
    h_new = int(100 * h / max(w, h) )

    img_resized = cv2.resize(masked_image, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    img_resized = img_resized.reshape((img_resized.shape[0] * img_resized.shape[1], 3))
    
    downsampled_image = np.array([])
    #R channels
    index = np.where(img_resized[:, 0] != -1.)
    downsampled_image = np.expand_dims(np.append(downsampled_image, img_resized[index[0], 0]), axis=-1)
    #G channels
    index = np.where(img_resized[:, 1] != -1.)
    downsampled_image = np.append(downsampled_image, np.expand_dims(img_resized[index[0], 1], axis=-1), axis=1)
    #B channels
    index = np.where(img_resized[:, 2] != -1.)
    downsampled_image = np.append(downsampled_image, np.expand_dims(img_resized[index[0], 2], axis=-1), axis=1)
    
    return masked_image, downsampled_image

def get_average_color(image, zipped):
    
    cluster_center_0 = zipped[0][1]
    cluster_center_1 = zipped[1][1]
    cluster_center_2 = zipped[2][1]
    
    #define average color
    average_color = np.zeros(image.shape)
    
    index = int(image.shape[1]/3.)
    
    average_color[:,0:index,:] = cluster_center_0
    average_color[:,index:index*2,:] = cluster_center_1
    average_color[:,index*2:index*3:] = cluster_center_2
        
    return average_color


def normalize_colors(image, prediction, clt):

    index = np.where(prediction == 0)
    
    #mask the non-relevant part of the image
    masked_image = image.astype('float')
    masked_image[index] = 0.

    ratio = 100
    
    # Resize image
    h, w, _ = masked_image.shape
    w_new = int(ratio * w / max(w, h) )
    h_new = int(ratio * h / max(w, h) )
    
    img_resized = cv2.resize(masked_image, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    img_resized = img_resized.reshape((img_resized.shape[0] * img_resized.shape[1], 3))
        
    out_list = []
    
    for i in range(len(clt.labels_)):
        out_list.append(clt.cluster_centers_[clt.labels_[i]])
    out_list = np.asarray(out_list)
        
    index = np.where(img_resized[:, 0] != 0.)
    
    img_resized[index] = out_list
    img_resized = img_resized.reshape((h_new, w_new, 3))

#    print(img_resized.shape)
    
    return img_resized

