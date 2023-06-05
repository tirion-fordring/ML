import numpy as np
from PIL import ImageDraw, Image


def readResults(file_path):
    points = []
    with open(file_path) as f:
        for line in f:
            line = line.split(',')[:-1]
            line = [int(t) for t in line]
            line = np.array(line).reshape(int(len(line) / 2), 2)
            xmin = np.min(line[:, 0])
            ymin = np.min(line[:, 1])
            xmax = np.max(line[:, 0])
            ymax = np.max(line[:, 1])
            points.append((xmin, ymin, xmax, ymax))
            break
    return points


if __name__ == '__main__':
    p = readResults('./demo_results/res_3.txt')
    img = Image.open('./3.jpg')
    draw = ImageDraw.Draw(img)
    draw.rectangle(p[0], outline='black', width=1)
    img.show()
    width = p[0][2] - p[0][0]
    height = p[0][3] - p[0][1]
    result_image = Image.new('RGB', (width, height), (0, 0, 0, 0))
    rect_on_big = img.crop(p[0])
    result_image.paste(rect_on_big)
    result_image.show()
    print(p)
