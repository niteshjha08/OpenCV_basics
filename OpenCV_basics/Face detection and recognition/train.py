import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
def read_images(path):
    print(path)
    images=[]
    labels=[]
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, dtype=np.float64)
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        images.append(image)
        labels.append(nbr)
    return images,labels

def train(image):
    ############################ Defining image vector parameters ##########################
    img_height=200
    img_width=200
    n_images=len(image)

    ################## Creating flattened image_vector ####################################
    image_vectors=np.zeros([n_images,img_height*img_width],dtype=np.float64)  # 166x (243*320)
    for i in range(n_images):
        image_vectors[i,:]=np.array(image[i],dtype=np.float64).flatten()
        plt.subplot(2,4,i+1)
        plt.imshow(image[i],cmap="gray")
    plt.show()

    ######################## Creating mean face of training set ###############################
    mean_face=np.zeros((1,img_height*img_width),dtype=np.float64)           # 1 x (243*320)
    for i in range(n_images):
        mean_face=image_vectors[i]+mean_face
    mean_face=np.divide(mean_face,n_images)
    plt.imshow(mean_face.reshape(img_height,img_width), cmap="gray")
    plt.title('mean_face')
    plt.show()

    ################# Normalizing image_vector (zeroing the data) ############################
    normalized_image_vectors=np.array((image_vectors-mean_face).round(),dtype=np.float64)                 # 166x (243*320)
    plt.imshow(normalized_image_vectors[0].reshape(img_height,img_width), cmap="gray")
    plt.title('normalized_face 1')
    plt.show()

    ########################### Finding covariance matrix of MxM dimension ####################
    ### As (height*width) is large, covariance of that matrix would be huge. (Complete the explanation) ###
    covariance_initial=np.cov(normalized_image_vectors,)                    # 166x 166
    covariance_initial=np.divide(covariance_initial,8.0)
    eigenvalues_initial, eigenvectors_initial = np.linalg.eig(covariance_initial)

    ##### Arranging in decreasing eigen value order, and then selecting principal components ####
    couple=[(eigenvalues_initial[i],eigenvectors_initial[:,i]) for i in range(len(eigenvalues_initial))]
    couple.sort(reverse=True)
    eigenvalues_initial_ordered=[couple[i][0] for i in range(len(couple))]
    print(eigenvalues_initial_ordered)
    eigenvectors_initial_ordered=[couple[i][1] for i in range(len(couple))]

    principal_eigenvalues_initial=eigenvalues_initial_ordered[:]
    principal_eigenvectors_initial=eigenvectors_initial_ordered[:]
    principal_eigenvectors_initial=(np.array(principal_eigenvectors_initial)).transpose()

    ##### Finding actual eigenvector by premultiplying with previous eigen vector. Refer Turk, Pentland paper for info #####
    eigenvectors_actual=np.dot(normalized_image_vectors.transpose(),principal_eigenvectors_initial)
    #print("Shape of actual eigen vector matrix:",eigenvectors_actual.shape)

    #### Creating Eigen faces by simply reshaping ########

    for i in range(len(principal_eigenvalues_initial)):
        eigenface_vector=np.array([eigenvectors_actual[:,i]])
        eigenface=eigenface_vector.reshape(img_height,img_width)
        plt.subplot(2,4,i+1)
        plt.imshow(eigenface,cmap="gray")
    plt.suptitle("Eigen faces")
    plt.show()

    #### Finding weights which give the linear combination coefficients for describing training faces using eigen faces #####
    weights = np.dot(normalized_image_vectors , eigenvectors_actual)
    print(weights.shape)

    #### Reconstructing trial for training images #####
    plt.figure(2)
    # for i in range(n_images):
    wts = np.array(weights.transpose())
    wts = np.divide(wts, wts.max())
    print("shape of eigenvectors_actual is", eigenvectors_actual.shape)
    print("shape of wts is ", wts.shape)
    reconstructed_face = np.dot(eigenvectors_actual, wts)

    for i in range(n_images):
        plt.subplot(2, 4, i + 1)
        total_face = np.add(reconstructed_face[:, i], 1 * mean_face)
        plt.imshow(total_face.reshape(img_height, img_width), cmap="gray")
    plt.suptitle('Reconstructed faces')
    plt.show()

    return weights

if __name__=="__main__":
    img_dir='training_set'
    path=os.path.join(os.getcwd(),img_dir)
    imgarray,labels=read_images(path)
    wts,lab=train(imgarray)


