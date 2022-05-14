import torch
from tqdm.notebook import tqdm

def train(device, model, num_epochs, optimizer, criterion, train_dl, val_dl):
    for epoch in tqdm(range(num_epochs)):
        loss_ep = 0

        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        print(f"Loss in epoch {epoch} :::: {loss_ep / len(train_dl)}")

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )