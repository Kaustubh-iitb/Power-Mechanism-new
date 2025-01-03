import numpy as np
import pandas as pd
# import dcMinMaxFunctions as dc
# import dcor
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import wandb

from churn_help import *
import time
import argparse

parser = argparse.ArgumentParser(description='Churn model private testing')
parser.add_argument('--data_path', type=str, default='data/Churn_Modelling.csv',
                    help='Path to the CSV file containing the Churn data')
parser.add_argument('--eps', type=float, default=1.0,
                    help='Set epsilon for the model')
parser.add_argument('--model_path', type=str, default='Models/churn_new_1',
                    help='Path to the Model to create embeddings')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training the model')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.003,
                    help='Learning rate for the optimizer')
parser.add_argument('--wandb_project', type=str, default='churn mod test',
                    help='Name of the Weights & Biases project to log metrics to')
parser.add_argument('--norm',type=float,default= 1,
                    help='Normalizing the data by multiplying with this number')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
args = parser.parse_args()

# You can access the parsed arguments like this:
data_path = args.data_path
eps = args.eps
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
wandb_project = args.wandb_project
model_path = args.model_path
norm = args.norm
net_depth = args.net_depth
 # adds all of the arguments as config variables
def main(data_path ,batch_size,num_epochs,learning_rate,model_path):
    X,Y = churn_data_loader(data_path,norm=norm)
    max_dist = torch.cdist(X, X).max()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_priv = torch.utils.data.TensorDataset(X_train, Y_train)
    test_priv = torch.utils.data.TensorDataset(X_test, Y_test)

    trainloader_priv = torch.utils.data.DataLoader(train_priv, batch_size=1000,
                                          shuffle=False, num_workers=2)
    testloader_priv = torch.utils.data.DataLoader(test_priv, batch_size=1000,
                                          shuffle=False, num_workers=2)

   
      
    # net = torch.load("../Code/Models/net_1_cov")
    state_dict = torch.load(model_path)

    # Create an instance of Net
    net = Net(net_depth)
    net.load_state_dict(state_dict)

    # net = net.to(torch.device('cuda'))
    # # Load the state dictionary into the model
    # net.load_state_dict(state_dict)
    # outp = net(X[0:2])
    # print(X[0:2])
    # print(outp)
    # print(net.y)

    

    
    X_emb_train,losses_train = create_model_embs2(net,trainloader_priv,device= torch.device('cuda'),l=len(X_train),h=0.65)
    X_emb_test,losses_test = create_model_embs2(net,testloader_priv,device= torch.device('cuda'),l=len(X_test),h=0.65)
    losses_train,indices = torch.sort(losses_train*max_dist)

    # print(indices)
    
     
    X_train = X_train[indices]
    X_emb_train = X_emb_train[indices]
    Y_train = Y_train[indices]
 
    run = wandb.init(project=wandb_project)
    wandb.config.update(args)
   
    set_eps = eps
    ind = (losses_train<set_eps).sum()
    print(ind)
    # num_epochs_eps = int(len(X)*num_epochs/ind)
    batch_size_eps = batch_size

    #write code for train test split using X_emb and Y

# Remove all things from cuda that were generated till now
    torch.cuda.empty_cache()
    
    
    
    
    train_emb_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_emb_train[0:ind],Y_train[0:ind]), batch_size=batch_size_eps,
                                            shuffle=False, num_workers=2,drop_last=True)
    test_emb_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_emb_test,Y_test), batch_size=batch_size_eps,
                                            shuffle=False, num_workers=2)
    
    #write code to append Xemb and Xemb_rest
    
    criterion = nn.BCELoss()
    # model = nn.Sequential(
    #             nn.Linear(54, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 7),
    #             nn.Softmax(dim=1)

    #         )
    model = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()

        )
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    time_start = time.time()
    train_emb(model, train_emb_loader, criterion, optimizer, num_epochs=num_epochs,device=torch.device('cuda'),test_loader = test_emb_loader,test_total_loader = None)
    time_end = time.time()
    print("Time taken to train the model: ",time_end-time_start)
    model.to(torch.device('cpu'))
    # test_model(model,test_emb_loader)
    # test_model(model,test_emb_full_loader)
if __name__ == "__main__":
    main(data_path=data_path,batch_size=batch_size,num_epochs=num_epochs,learning_rate=learning_rate,model_path=model_path)

