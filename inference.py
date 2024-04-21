import torch
import os
from models.yolo import YOLO
from PIL import Image, ImageDraw
from DataProcess import transforms
from models.vgg import get_vgg_backbone


def inference(model, img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for file_name in os.listdir(img_path):
        src_image = Image.open(img_path + file_name)
        width, height = src_image.width, src_image.height

        image = transforms(src_image).unsqueeze(0).numpy() / 255.0
        output = model(torch.Tensor(image).to(device)).view(49, 10)
        output = output.detach().cpu().numpy()

        threshold = 0.5
        candidates = list()

        for cell_idx in range(49):
            row, col = cell_idx // 7, cell_idx % 7
            r = output[cell_idx]
            bbox1 = r[:4]
            bbox1_confidence = r[4]
            if bbox1_confidence > threshold:
                candidates.append(((row, col), bbox1, bbox1_confidence))

            bbox2 = r[5:9]
            bbox2_confidence = r[9]
            if bbox2_confidence > threshold:
                candidates.append(((row, col), bbox2, bbox2_confidence))

        draw = ImageDraw.Draw(src_image)

        for item in candidates:
            (row, col), bbox, confidence = item
            width_per_cell = width / 7
            height_per_cell = height / 7

            x1 = (row + bbox[0]) * width_per_cell - 0.5 * width * bbox[2]
            x2 = (row + bbox[0]) * width_per_cell + 0.5 * width * bbox[2]
            y1 = (col + bbox[1]) * height_per_cell - 0.5 * height * bbox[3]
            y2 = (col + bbox[1]) * height_per_cell + 0.5 * height * bbox[3]

            draw.rectangle((x1, y1, x2, y2), outline=(0,255,0), width=3)

        src_image.show()


backbone = get_vgg_backbone()
model = YOLO(backbone)
path = 'images/'
model.load_state_dict(torch.load("checkpoints/yolo135.pt"))
inference(model, path)