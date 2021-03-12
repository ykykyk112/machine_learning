import torch
from model import cnn
from sklearn.metrics import confusion_matrix
import numpy as np

def get_all_pred(model, loader) :
    model.eval()

    is_cuda = torch.cuda.is_available()

    all_pred = torch.Tensor([])
    all_label = torch.Tensor([])
    for batch in loader :
        image, label = batch
        # Load model, data on GPU
        if is_cuda :
            device = torch.device('cuda')
            image, label = image.to(device), label.to(device)
            model = model.to(device)
            all_pred = all_pred.to(device)
            all_label = all_label.to(device)
        pred = model.forward(image)
        _, pred = torch.max(pred, dim = 1)
        all_pred = torch.cat((all_pred, pred), dim = 0)
        all_label = torch.cat((all_label, label), dim = 0)
    return all_pred, all_label

def remove_diag(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]) :
            if i == j :
                x[i][j] == 0
    return x

def get_cf_matrix(model, loader):
    pred, label = get_all_pred(model, loader)
    pred, label = pred.cpu().numpy(), label.cpu().numpy()
    cf_matrix = confusion_matrix(label, pred)
    return cf_matrix

def get_cf_removed(model, loader):
    pred, label = get_all_pred(model, loader)
    pred, label = pred.cpu().numpy(), label.cpu().numpy()
    cf_matrix = confusion_matrix(label, pred)
    ret_matrix = remove_diag(cf_matrix)
    return ret_matrix