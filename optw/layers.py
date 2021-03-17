import torch
import torch.nn as nn
from optw.hopfield import HopfieldPooling

class Classifier(nn.Module):

    def __init__(self, input_size, dropout=0.2):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_size, 2)
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)

class MLP(nn.Module):

    def __init__(self, input_size, hidden_layers = 512, dropout=0.2):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers, hidden_layers//2),
            nn.ReLU(),
            nn.Linear(hidden_layers//2, 2)
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)


class LSTM(nn.Module):

    def __init__(self, ftr_size, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(1, ftr_size, 1, batch_first=False)
        self.classifier = Classifier(ftr_size, dropout)
        self.ftr_size = ftr_size

    def create_hiddens(self, x):
        
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            device = x.device
        else:
            batch_size = x[0].shape[0]
            device = x[0].device

        hiddens = (torch.zeros(1, batch_size, self.ftr_size, device=device ),  
            torch.zeros(1, batch_size, self.ftr_size, device=device)) 
        
        return hiddens

    def forward(self, x) -> torch.Tensor:

        hiddens = self.create_hiddens(x)
        _ , hiddens = self.lstm(x.reshape((x.shape[1], -1, 1)), hiddens)
        return self.classifier(hiddens[0]).squeeze()




class Convolutional(nn.Module):
    def __init__(self, dropout=0.2):
        super(Convolutional, self).__init__()

        self.layer = nn.Sequential(
            nn.AvgPool1d(kernel_size=9, stride = 4),
            nn.Conv1d(1, 64, kernel_size=9, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=9, stride = 4),
            nn.Dropout(p=dropout),
            nn.Conv1d(64, 128, kernel_size=9, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=9, stride = 4),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, 256, kernel_size=9, stride=1),
            nn.AvgPool1d(kernel_size=9, stride = 4),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 512, kernel_size=9, stride=1),
            # nn.AvgPool1d(kernel_size=25), #global avg pool
        )
        self.hop = HopfieldPooling(input_size = 26, hidden_size = 16, quantity = 14)
        self.classifier = Classifier(364)
        # self.classifier = Classifier(512)


    def forward(self, x) -> torch.Tensor:

        x = x.reshape((-1, 1, x.shape[1]))
        x = self.layer(x)
        x = x.squeeze()
        x = self.hop(x)
        return  self.classifier(x)
        #return self.layer(x).squeeze()

