import os
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image


def process(image_dir, annot_dir):
    transforms = T.Compose([
        T.PILToTensor(),
        T.Resize((224, 224))
    ])

    files = list(os.listdir(image_dir))
    label = np.zeros((len(files), 7, 7, 5))
    feat = np.zeros((len(files), 3, 224, 224))

    for idx, file in enumerate(tqdm(files)):
        annot = file.replace('.jpg', '.txt')

        ret = []

        with open(annot_dir + annot, 'r') as f:
            for content in f.readlines():
                content = content.replace('Human face', '')

                ret.append((float(x) for x in content.split()))

        img = Image.open(image_dir + file)
        img = img.convert('RGB')

        img_height = img.height
        img_width = img.width

        feat[idx] = transforms(img).numpy() / 255.0

        height_per_cell = img_height / 7
        width_per_cell = img_width / 7

        for (x0, y0, x1, y1) in ret:
            x = (x0 + x1) / 2.0
            y = (y0 + y1) / 2.0
            w = (x1 - x0) / img_width
            h = (y1 - y0) / img_height

            x_cell = int(x / width_per_cell)
            y_cell = int(y / height_per_cell)

            x_in_cell = (x - x_cell * width_per_cell) / width_per_cell
            y_in_cell = (y - y_cell * height_per_cell) / height_per_cell

            offset = 0

            label[idx][x_cell][y_cell][offset + 0] = x_in_cell
            label[idx][x_cell][y_cell][offset + 1] = y_in_cell
            label[idx][x_cell][y_cell][offset + 2] = w
            label[idx][x_cell][y_cell][offset + 3] = h
            label[idx][x_cell][y_cell][offset + 4] = 1.0

    return feat, label


feat, label = process('data/images/train/', 'data/labels2/')
np.save("../FaceDetect/feat.npy", feat)
np.save("../FaceDetect/label.npy", label)