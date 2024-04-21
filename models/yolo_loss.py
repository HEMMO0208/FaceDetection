import torch


def get_iou(pred, true):
    if true[2] == 0 or true[3] == 0:
        return 0

    area_1 = pred[2] * pred[3] * 49
    area_2 = true[2] * true[3] * 49

    pred[2] *= 3.5
    pred[3] *= 3.5
    true[2] *= 3.5
    true[3] *= 3.5

    x1 = max(pred[0] - pred[2], true[0] - true[2])
    x2 = min(pred[0] + pred[2], true[0] + true[2])
    y1 = max(pred[1] - pred[3], true[1] - true[3])
    y2 = min(pred[1] + pred[3], true[1] + true[3])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)

    return intersection / (area_1 + area_2 - intersection)


def yolo_loss(y_pred, y_true):
    batch_loss = None
    batch_size = len(y_true)

    for batch in range(batch_size):
        y_true_unit = y_true[batch]
        y_pred_unit = y_pred[batch]
        y_true_unit = y_true_unit.view(7 * 7, 5)
        y_pred_unit = y_pred_unit.view(7 * 7, 10)

        loss = None

        for cell in range(len(y_true_unit)):
            pred1 = y_pred_unit[cell, :4]
            confidence1 = y_pred_unit[cell, 4]
            pred2 = y_pred_unit[cell, 5:9]
            confidence2 = y_pred_unit[cell, 9]

            bbox_true = y_true_unit[cell, :4]
            bbox_true_numpy = bbox_true.detach().cpu().numpy()
            true_confidence = y_true_unit[cell, 4]

            if bbox_true_numpy[2] == 0 or bbox_true_numpy[3] == 0:
                if loss is None:
                    loss = 0.5 * (torch.square(confidence1 - true_confidence) + torch.square(confidence2 - true_confidence))
                else:
                    loss += 0.5 * (torch.square(confidence1 - true_confidence) + torch.square(confidence2 - true_confidence))
                continue

            iou_1 = get_iou(pred1.detach().cpu().numpy(), bbox_true_numpy)
            iou_2 = get_iou(pred2.detach().cpu().numpy(), bbox_true_numpy)

            if iou_1 >= iou_2:
                responsible = pred1
                responsible_confidence = confidence1
                not_responsible = confidence2

            else:
                responsible = pred2
                responsible_confidence = confidence2
                not_responsible = confidence1

            res_hw = torch.sqrt(responsible[2:])
            real_hw = torch.sqrt(bbox_true[2:])

            localize_loss_1 = torch.square(bbox_true[0] - responsible[0]) + torch.square(bbox_true[1] - responsible[1])
            localize_loss_2 = torch.square(res_hw[0] - real_hw[0]) + torch.square(res_hw[1] - real_hw[1])

            if torch.isnan(localize_loss_1):
                localize_loss_1 = 0

            if torch.isnan(localize_loss_2):
                localize_loss_2 = 0

            confidence_loss = torch.square(responsible_confidence - true_confidence) + torch.square(not_responsible)

            cell_loss = 5.0 * (localize_loss_1 + localize_loss_2) + confidence_loss

            if loss is None:
                loss = cell_loss
            else:
                loss += cell_loss

        if batch_loss is None:
            batch_loss = loss
        else:
            batch_loss += loss

    batch_loss /= batch_size

    return batch_loss
