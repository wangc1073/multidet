
import torch
#######################################################
def mape_loss(true,pre):
    dataabs = torch.abs(true - pre)
    return torch.mean(dataabs / true)
#######################################################
def rmse_loss(true,pre):
    # RMSE = np.sqrt(np.mean(np.square(true - pre))) 
    # .pow(2)
    RMSE = (pre - true).norm(2)
    return RMSE
#######################################################




#
def pinball_loss(q,labels,out):
    # labels [batch,3,24]
    labels = labels.reshape([-1, 1])
    out = out.reshape([-1, 1])
    lossout = 0
    for count in range(labels.shape[0]):
        if labels[count] > out[count]:
            lossout = lossout + q*(labels[count] - out[count])
        else:
            lossout = lossout + (1-q)*(out[count] - labels[count])
    return lossout
############################


def picp_loss(pre,upper,lower):
    # labels [batch,3,24]
    pre = pre.reshape([-1, 1])
    upper = upper.reshape([-1, 1])
    lower = lower.reshape([-1, 1])

    picp = 0

    for count in range(pre.shape[0]):

        if pre[count] < upper[count] and pre[count] > lower[count]:
            c = 1
        else:
            c = 0
        ########
        picp = picp + c

    picp = picp / ( 1.0 * pre.shape[0] )
    return picp
############################

def ace(picp, pinc):

    return picp - pinc
############################