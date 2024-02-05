from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import datetime
from lossFunc import mape_loss, rmse_loss
from multidet import MultiDeT
import os

# You need to customize the ReadData function here
from readData import loadDataforTrainVal

# GPU?
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_x_data, _y_data = loadDataforTrainVal()
#########################################################

batch_size = 32
train_dataset = TensorDataset(_x_data, _y_data)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

net = MultiDeT()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

num_epoch = 200
for epoch in range(num_epoch):

    num_batch = len(train_loader)
    for i, data in enumerate(train_loader):
        start_time = datetime.datetime.now()

        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()

        optimizer.zero_grad()

        label1 = torch.flatten(input=labels[:, :, : 1], start_dim=1)
        label2 = torch.flatten(input=labels[:, :, 1: 2], start_dim=1)
        label3 = torch.flatten(input=labels[:, :, 2: 3], start_dim=1)

        output1, output2, output3 = net(src=inputs)

        loss1 = mape_loss(label1, output1)
        loss2 = mape_loss(label2, output2)
        loss3 = mape_loss(label3, output3)
        loss = loss1 * 0.5 + loss2 * 0.2 + loss3 * 0.3

        loss.backward()
        optimizer.step()

        end_time = datetime.datetime.now()

        lasttime = (end_time - start_time) * (num_batch - i) + (end_time - start_time) * num_batch * (
                num_epoch - epoch - 1)
        print(" eta: ", lasttime,
              " epoch: %4d in %4d, batch: %5d  loss: %.4f  loss1: %.4f  loss2: %.4f  loss3: %.4f" % (
                  epoch + 1, num_epoch, (i + 1), loss.item(), loss1.item(), loss2.item(), loss3.item()))

torch.save(net, 'MultiDet.pt')
print('Finished Training')
####################
