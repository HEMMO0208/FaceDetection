from tqdm import tqdm
import torch


def train_yolo(model, criterion, optimizer, epochs, data_loader, device):
    pbar = tqdm(range(epochs), desc="training", mininterval=0.01)

    for epoch in pbar:
        if 0 <= epoch < 75:
            optimizer.param_groups[0]['lr'] = 0.001 + 0.009 * (float(epoch)/(75.0))

        elif 75 <= epoch < 105:
            optimizer.param_groups[0]['lr'] = 0.001

        else:
            optimizer.param_groups[0]['lr'] = 0.0001

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar_str = "training, [loss = %.4f]" % loss.item()
            pbar.set_description(pbar_str)

        torch.save(model.state_dict(), "yolo%d.pt" % (epoch + 1))