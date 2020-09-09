from train import read_images, train
import os
def test(dir):

    #get mean_face, eigenvectors_actual,weights,labels
    path = os.path.join(os.getcwd(), dir)
    test_images,_=read_images(path)
    print(len(test_images))
    image_vector=np.array(test_images,dtype=np.float64).flatten()
    normalized_test = image_vector - mean_face
    w_unknown = np.dot(normalized_test, eigenvectors_actual)
    sse = np.zeros(len(weights))
    for i in range(len(weights)):
        sse[i] = ((weights[i] - w_unknown).sum()) ** 2
    index=np.where(sse==np.amin(sse))
    print("Face detected is:",labels[index])

if __name__=="__main__":
    img_dir = 'training_set'
    path = os.path.join(os.getcwd(), img_dir)
    imgarray, labels = read_images(path)
    weightsqq=train(imgarray)
    test_img_dir='testing_set'
    test(test_img_dir)
