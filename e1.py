from googlenet import create_googlenet
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

def preprocess_image(image):
    image = image.resize((224, 224))
    # image.show()
    image = img_to_array(image)
    image = imagenet_utils.preprocess_input(image)
    image = np.expand_dims(image, axis = 0)
    return image

if __name__ == '__main__':
    images = []
    for index in range(1049, 1125):
        image = Image.open(f'tvsum_data/tvsum_data/frame{index}.jpg')
        image = preprocess_image(image)
        images.append(image)

    model = create_googlenet('googlenet_weights.h5')

    labels = np.loadtxt('synset_words.txt', str, delimiter = '\t')

    outputs = [model.predict(image) for image in images]

    for output in outputs:
        predicted_label = np.argmax(output[3])
        predicted_class_name = labels[predicted_label]
        print('Predicted Class: ', predicted_label, ', Class Name: ', predicted_class_name)

    # for index in range(3):
    #     print('Cosine similarity 1, 2: ', cosine_similarity(outputs[0][index + 1], outputs[1][index + 1]))
    #     print('Cosine similarity 1, 3: ', cosine_similarity(outputs[0][index + 1], outputs[2][index + 1]))
    #     print('Cosine similarity 1, 4: ', cosine_similarity(outputs[0][index + 1], outputs[3][index + 1]))
    #     print('Cosine similarity 2, 3: ', cosine_similarity(outputs[1][index + 1], outputs[2][index + 1]))
    #     print('Cosine similarity 2, 4: ', cosine_similarity(outputs[1][index + 1], outputs[3][index + 1]))
    #     print('Cosine similarity 3, 4: ', cosine_similarity(outputs[2][index + 1], outputs[3][index + 1]))

    for index in range(3):
        embeddings = []
        for output in outputs:
            embeddings.append(output[index + 1][0])
        pca = PCA(n_components = 2)
        reduced_embeddings = pca.fit_transform(embeddings)

        scenes = [reduced_embeddings]
        for scene in scenes:
            print(len(scene))

        plt.figure()
        for scene_index, scene in enumerate(scenes):
            plt.scatter([frame[0] for frame in scene], [frame[1] for frame in scene], label = f'scene {scene_index + 1}')
        plt.legend()
    plt.show()
