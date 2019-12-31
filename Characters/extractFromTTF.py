import cv2
import numpy
from PIL import Image, ImageFont, ImageDraw
import string
import os


def get_stock_characters():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'Kenteken.ttf')
    font = ImageFont.truetype(filename, 80)
    numbers = string.digits
    characters = numbers + string.ascii_uppercase
    stock_characters = []
    stock_numbers = []
    for c in characters:
        im = Image.new("RGB", (800, 600))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, font=font)
        im = im.crop(im.getbbox())
        im = numpy.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
        stock_characters.append(im)
        if c in numbers:
            stock_numbers.append(im)
    return stock_characters, stock_numbers


get_stock_characters()
