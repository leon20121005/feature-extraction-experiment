from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def get_red_code(value):
    return '#%02x%02x%02x' % (value, 0, 0)

def get_green_code(value):
    return '#%02x%02x%02x' % (0, value, 0)

def get_blue_code(value):
    return '#%02x%02x%02x' % (0, 0, value)

if __name__ == '__main__':
    images = []

    for index in range(218, 242):
        image = Image.open(f'tvsum_data/tvsum_data/frame{index}.jpg')
        images.append(image)

    histograms = []

    for image in images:
        histogram = image.histogram()
        histograms.append(histogram)

    histograms = [histograms[0], histograms[5], histograms[10], histograms[12], histograms[17], histograms[22]]

    for histogram in histograms:
        plt.figure()
        # red histogram
        for index in range(0, 256):
            plt.bar(index, histogram[index], color = get_red_code(index), edgecolor = get_red_code(index), alpha = 0.3)
        # green histogram
        for index in range(256, 512):
            plt.bar(index, histogram[index], color = get_green_code(index - 256), edgecolor = get_green_code(index - 256), alpha = 0.3)
        # blue histogram
        for index in range(512, 768):
            plt.bar(index, histogram[index], color = get_blue_code(index - 512), edgecolor = get_blue_code(index - 512), alpha = 0.3)
        print('Total length: ', np.sum(histogram))
    plt.show()

    # print('Cosine similarity 1, 2: ', cosine_similarity([histograms[0]], [histograms[1]]))
    # print('Cosine similarity 1, 3: ', cosine_similarity([histograms[0]], [histograms[2]]))
    # print('Cosine similarity 1, 4: ', cosine_similarity([histograms[0]], [histograms[3]]))
    # print('Cosine similarity 2, 3: ', cosine_similarity([histograms[1]], [histograms[2]]))
    # print('Cosine similarity 2, 4: ', cosine_similarity([histograms[1]], [histograms[3]]))
    # print('Cosine similarity 3, 4: ', cosine_similarity([histograms[2]], [histograms[3]]))

    # print('Euclidean distance 1, 2: ', euclidean_distances([histograms[0]], [histograms[1]]))
    # print('Euclidean distance 1, 3: ', euclidean_distances([histograms[0]], [histograms[2]]))
    # print('Euclidean distance 1, 4: ', euclidean_distances([histograms[0]], [histograms[3]]))
    # print('Euclidean distance 2, 3: ', euclidean_distances([histograms[1]], [histograms[2]]))
    # print('Euclidean distance 2, 4: ', euclidean_distances([histograms[1]], [histograms[3]]))
    # print('Euclidean distance 3, 4: ', euclidean_distances([histograms[2]], [histograms[3]]))
