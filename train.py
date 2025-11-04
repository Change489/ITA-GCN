import torch
import torch.nn as nn


def train_model_process(model, train_dataset, config, store_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss()


    best_loss = float("inf")


    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0

        model.train()
        for data in train_dataset:

            optimizer.zero_grad()
            train_data = data[0].to(device)
            train_label = data[1].to(device)

            output = model.forward(train_data)

            loss = criterion(output, train_label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * train_label.size(0)

        epoch_loss /= len(train_dataset)


        long_string = f"epoch {epoch+1} train loss:{epoch_loss:.5f}"


        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), store_path)
            long_string += " --> Best model ever (stored)"

        print(long_string)








