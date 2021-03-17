import torch
import torch.nn as nn

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
    def __init__(self, input_size, dropout=0.2):
        super(Convolutional, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.Linear(input_size, hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers, hidden_layers//2),
            nn.ReLU(),
            nn.Linear(hidden_layers//2, 2)
        )



    def forward(self, x) -> torch.Tensor:
        return self.layers(x).squeeze()

