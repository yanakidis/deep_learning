import torch
import matplotlib.pyplot as plt

def train(model, train_data, val_data, optimizer, epochs, max_length, batch_size, Dataset, logs=False):
    train, val = Dataset(train_data, max_length), Dataset(val_data, max_length)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        if logs:
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                    | Val Loss: {total_loss_val / len(val_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_data): .3f}')

def evaluate(model, test_data, max_length, batch_size, Dataset, logs=False):
    test = Dataset(test_data, max_length)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    if logs:
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

def get_plot(acc_dict, title, xlabel, xticks=None):
    right_order = sorted([(x,y) for x,y in zip(acc_dict.keys(),acc_dict.values())])
    x = [x for x,y in right_order]
    y = [y for x,y in right_order]

    plt.figure(figsize = (8,4))
    plt.title(title)
    plt.plot(x, y, color='g', marker='o')
    plt.ylabel('качество')
    plt.xlabel(xlabel)
    if xticks is not None:
    plt.xticks(x, xticks)
    plt.grid(True)
    plt.show()