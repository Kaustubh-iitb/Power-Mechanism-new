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



class Net(nn.Module):
    def __init__(self,p):
        super(Net, self ).__init__()
        
        self.loss_reg = 0
        self.p =p 
        self.x = 0
        self.y = 0
        self.H_net1 = nn.Sequential(
            nn.Linear(54, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 54*54).cuda()
        )
        self.X_net = nn.Sequential(
            nn.Linear(54, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(dim=2)

        )
        
    def forward(self, x):
        def H_mul(z):
            H12 = self.H_net1(z)
            H12= H12.reshape(z.shape[0],d,d)
            x12 = torch.matmul(z,H12)
            return(x12)
    
        
        def batch_jacobian(func, z, create_graph=False):
            # x in shape (Batch, Length)
            def _func_sum(z):
                return func(z).sum(dim=0)
            return torch.squeeze(torch.autograd.functional.jacobian(_func_sum, z, create_graph=create_graph)).permute(1,0,2)
        
        x.requires_grad =True
        p = self.p
        self.x = x
        d = x.shape[1]
        bs = x.shape[0]
        x= torch.unsqueeze(x,1)
        z = x.cuda()
        loss_reg = torch.zeros(bs,d).cuda()
        for i in range(p):
            H = self.H_net1(z).cuda()
            H = H.reshape(bs,d,d)
            z = torch.matmul(z,H).cuda()
            J = batch_jacobian(H_mul, z, create_graph=True)
            J_int =-torch.log(torch.abs(torch.det(J)))
            loss_reg = loss_reg + torch.squeeze(torch.autograd.grad(J_int, x,torch.ones_like(J_int),allow_unused=True,create_graph= True)[0]).cuda()
        self.loss_reg = loss_reg
        self.y = z
        y = self.X_net(z)
        return y

def gau_ker(u):
    return torch.pow(2*torch.tensor(torch.pi),u.shape[1]/(-2))*torch.exp(torch.bmm(u.view(u.shape[0], 1, u.shape[1]), u.view(u.shape[0],  u.shape[1],1))/(-2)).cuda()


def py_kde(x,X_t,h):
    norm = X_t.shape[0]*(h**x.shape[1])
    prob = torch.zeros(x.shape[0]).cuda()
    for i in range(len(X_t)):
        prob+= (torch.squeeze(gau_ker((x - X_t[i])/h))/norm).cuda()
    return(prob)


def py_kde_der(p_x,x):
    # x.requires_grad = True
    # p_x = py_kde(x,X_t,h)
    return (torch.autograd.grad(p_x,x,torch.ones_like(p_x),allow_unused=True,create_graph=True)[0]).cuda()


def gau_ker_der(X,h):
    N= X.shape[0]
    d = X.shape[1]
    grad = torch.zeros(X.shape)
    for n in range(N):
        for i in range(d):
            for j in range(N):
                grad[n][i]+= torch.exp(-1*torch.dot((X[n]-X[j]),(X[n]-X[j]))/(2*h*h))*(X[n][i] -X[j][i]) /(N*(h**(d+2))*((2*math.pi)**(d/2)))

    return grad


def CI_KDE(p_x,n,h,d,alpha):
    return( stats.norm.ppf(1-alpha/2)*torch.sqrt(p_x/((2**d)*math.sqrt(torch.pi**d)*n*h**(d))).cuda() )

def CI_KDE_der(p_x_der,p_x,n,h,d,alpha):
    return( p_x_der*stats.norm.ppf(1-alpha/2)*torch.sqrt(1/(p_x.unsqueeze(dim=1)*(2**d)*math.sqrt(torch.pi**d)*n*h**(d))).cuda() )



def normalize(x):
    x_normed = 0.1*x /(x.max(0, keepdim=True)[0])
    return x_normed

def cov_data_loader(path):
    """
    This function loads the data from the path and returns the data as a pandas dataframe.
    """
    df = pd.read_csv(path)
    X = df.drop("Cover_Type",axis=1).values
    Y= df["Cover_Type"].values
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    X = normalize(X)
    #Write code to convert Y from 1,2,3,4,5,6,7 to 0,1,2,3,4,5,6
    Y = Y-1
    Y = Y.type(torch.LongTensor)
    X = X[::10]
    Y= Y[::10]


    return X,Y


def train_model_priv(net,trainloader,optimizer,epochs,h,rate=10,device= torch.device('cpu'),print_cond = True,only_reg_flag=0,lr_schedular =None):
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    lr = lr_schedular
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):  # loop over the dataset multiple times
        # scheduler.step()
        running_loss = 0.0
        running_loss_reg = 0.0
        if(lr):
       
            for groups in optimizer.param_groups:
                groups['lr'] = lr(epoch)
        # optimizer.param_groups[0]['lr'] = lr(epoch)
        
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs = data[0].to(device)
            inputs.requires_grad = True
            labels = data[1].to(device)
            f = py_kde(inputs,inputs,h)
            f_der = py_kde_der(f,inputs)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if(only_reg_flag==1):
                loss = torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum()
            elif(only_reg_flag==2):
                loss = criterion(torch.squeeze(outputs),torch.squeeze(labels))
                
            else:
                loss = 1000*criterion(torch.squeeze(outputs),torch.squeeze(labels)) + torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum()
            loss.backward(retain_graph=True)

            optimizer.step()
            loss = loss.detach().cpu()/len(inputs)

          

            loss_reg = torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum().detach().cpu()/len(inputs)
            wandb.log({"loss": loss.item(),"loss_reg":loss_reg.item()})

            # print statistics
            # print(loss.sum().shape)
            running_loss += loss.item()
            running_loss_reg += torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum().item()
            # if i % 100 == 99:    # print every 2000 mini-batches
            # if((i+1)%rate==0):

            #     if(print_cond):
                    
            #         print("Epoch: ",epoch + 1,"Loss: " ,running_loss /(rate*trainloader.batch_size),"Reg Loss: ",running_loss_reg /(rate*trainloader.batch_size))
            #         running_loss = 0.0
            #         running_loss_reg = 0.0


class LearnerRateScheduler:
    """Learning rate scheduler class. This class implements the learning rate scheduler."""
    def __init__(self, type, base_learning_rate, warmup_epochs=0, **kwargs):
        """summary

        Args:
            type (str): The type of the learning rate scheduler. Can be one of 'constant', 'linear', 'exponential' or 'step'.
            base_learning_rate (float): The learning rate to start with after warmup_epochs.
            warmup_epochs (int, optional): The number of epochs for warm-up. Linear in nature. Goes from lr_init(defaults to 0) to base learning rate . Defaults to 10.
            **kwargs: Additional arguments for the learning rate scheduler. The arguments depend on the type of scheduler used.
            
            
        Raises:
            TypeError: description
            TypeError: description
            TypeError: description
            TypeError: description
            TypeError: description
        """
        self.type = type
        self.base_learning_rate = base_learning_rate
        allowed_parameters = ['final_learning_rate', 'decay_rate', 'decay_steps', 'total_epochs', 'lr_init']
        # Check if any unknown keys are present in kwargs
        unknown_parameters = set(kwargs.keys()) - set(allowed_parameters)
        if unknown_parameters:
            raise TypeError(f"Unknown parameter(s) provided: {', '.join(unknown_parameters)}")
        self.final_learning_rate = kwargs['final_learning_rate'] if 'final_learning_rate' in kwargs else None
        self.decay_rate = kwargs['decay_rate'] if 'decay_rate' in kwargs else None
        self.decay_steps = kwargs['decay_steps'] if 'decay_steps' in kwargs else None
        self.warmup_epochs = warmup_epochs
        self.total_epochs = kwargs['total_epochs'] if 'total_epochs' in kwargs else None
        self.lr_init = kwargs['lr_init'] if 'lr_init' in kwargs else 0.0
        
        if self.type == 'linear':
            if 'final_learning_rate' not in kwargs.keys():
                raise TypeError(f"final_learning_rate must be provided for linear decay")
            if 'total_epochs' not in kwargs.keys():
                raise TypeError(f"total_epochs must be provided for linear decay")
        if self.type == 'step':
            if 'decay_rate' not in kwargs.keys():
                raise TypeError(f"decay_rate must be provided for step decay")
            if 'decay_steps' not in kwargs.keys():
                raise TypeError(f"decay_steps must be provided for step decay")
    
    def __call__(self, step):
        if step < self.warmup_epochs:
            #linear increase to base_learning_rate
            return self.lr_init + (self.base_learning_rate-self.lr_init) * (step / self.warmup_epochs)
        else:
            if self.type == 'constant':
                return self.base_learning_rate
            elif self.type == 'linear':
                return self.base_learning_rate - (self.base_learning_rate - self.final_learning_rate) * (step - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            elif self.type == 'exponential':
                pass
            elif self.type == 'step':
                return self.base_learning_rate * (self.decay_rate ** (int(step / self.decay_steps)))
            else:
                raise NotImplementedError

def create_model_embs(net,trainloader,device= torch.device('cpu'),l=0,h=0.82):
    alpha =1/l;
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    X_emb = torch.zeros(l,54)
    losses = torch.zeros(l)


    net = net.to(device)
    # criterion = nn.CrossEntropyLoss()

    bs = trainloader.batch_size
        
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
   
        inputs = data[0].to(device)
        n = len(inputs)
        d = inputs.shape[1]
        inputs.requires_grad = True
        labels = data[1].to(device)
        f = py_kde(inputs,inputs,h)


        f_der = py_kde_der(f,inputs)
      
        ci = CI_KDE(f,n,h,d,alpha)
       
        output =  net(inputs)
        
        loss =torch.max(torch.linalg.norm(f_der/(f-ci).view(f.shape[0],1)+net.loss_reg,dim=1),torch.linalg.norm(f_der/(f+ci).view(f.shape[0],1)+net.loss_reg,dim=1)) 
        try:
            losses[i*bs:i*bs+len(loss)] =loss.detach().cpu()
        except:
            print(loss.detach().cpu().shape)
            print(len(loss))
            print(net.y.detach().cpu().shape)
            print(i*bs)
            print(X_emb[i*bs:i*bs+len(loss)].shape)
        X_emb[i*bs:i*bs+len(loss)] = torch.squeeze(net.y.detach().cpu())
    return(X_emb,losses)

def train_emb(model, train_loader, loss_fn, optimizer, num_epochs,device=torch.device('cpu')):
    running_loss = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # wandb.log({"loss": loss.item()/len(inputs)})
            # # counter =0
        if((epoch+1)%10==0):
            print('Epoch [%d], loss: %.3f' % (epoch + 1, running_loss /(10* len(train_loader))))
            running_loss = 0.0


def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
 
