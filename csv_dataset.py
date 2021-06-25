import csv

import pandas
from PIL import Image
import numpy
from numpy import asarray

def save_dataset():
    image = Image.open("Untitled.png")
    image_sequence = asarray(image)

    writer = csv.writer(open('dataset.csv', 'w', newline=''))
    writer.writerows(image_sequence)

    data = numpy.genfromtxt('dataset.csv', dtype=int, delimiter=',')
    return data
