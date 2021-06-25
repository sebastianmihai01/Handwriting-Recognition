import csv
from PIL import Image
from numpy import asarray

def __init__():
    image = Image.open("untitled.png")
    image_sequence = asarray(image)
    print(image_sequence)

    writer = csv.writer(open('protagonist.csv', 'w', newline=''))
    writer.writerows(image_sequence)

__init__()
