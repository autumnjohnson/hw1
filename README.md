train_data, test_data = q1_sample_data_1()
train_losses, test_losses, distribution = q1_a(train_data, test_data, 20, 1)
softmax(train_data, axis=0)

log_softmax = nn.LogSoftmax(dim=1)
theta = torch.zeros(20, requires_grad=True)


loss = nn.NLLLoss(reduction='mean')
# input is of size N=batchsize x C=classes = 3 x 5


data = torch.utils.data.BatchSampler(
    train_data, batch_size=20, drop_last=False)
for batch in data:
    # each element in target has to have 0 <= value < C

    output = loss(-log_softmax(theta), torch.tensor(batch))
    output.backward()
output


def old_sdg(train_data, test_data, d, dset_id):
    theta = torch.zeros(d, dtype=torch.float32, requires_grad=True)
    train_losses = []
    test_losses = [loss_nll(torch.tensor(test_data), theta).item()]

    data = list(torch.utils.data.BatchSampler(
        train_data, batch_size=32, drop_last=False))

    optimizer = torch.optim.SGD([theta], lr=0.1)

    for _ in range(20):
        for batch in data:
            optimizer.zero_grad()
            batch = torch.tensor(batch)
            loss = loss_nll(batch, theta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_losses.append(loss_nll(torch.tensor(test_data), theta).item())

    return train_losses, test_losses, softmax(theta, dim=0).detach().numpy()