
# HMC MCMC with NUTS
import sys
import torch
from pyro.infer.mcmc.api import MCMC, NUTS
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
import pyro
import pyro.optim as optim
import numpy as np

class bay_reg_mcmc:
    def __init__(self,model, num_samples=1000, warmup_steps=100, cuda=False):
            
        self.model=model
        self.num_samples=num_samples
        self.warmup_steps=warmup_steps
        self.cuda = cuda

               
    def fit(self,x_train,y_train, verbose=False):
        if verbose:
            disable_progbar=False
        else:
            disable_progbar=True
        
        if self.cuda == True:
            x_train=torch.tensor(x_train).cuda()
            y_train=torch.tensor(y_train).cuda()   
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            x_train=torch.tensor(x_train)
            y_train=torch.tensor(y_train)
        
        self.posterior=MCMC(kernel=NUTS(self.model), num_samples=self.num_samples, warmup_steps=self.warmup_steps, disable_progbar=disable_progbar)
        
        self.posterior=self.posterior
        self.posterior.run(x_train,y_train)
        
    def predict(self, x_test, num_samples=300, percentiles=(5.0, 50.0, 95.0)):
        x_test=torch.tensor(x_test)
        samples=self.posterior.get_samples()
        #posterior_predictive=predictive(self.model, samples, x_test, num_samples=num_samples,return_sites=("w","b","_RETURN"))
        
        posterior_predictive= Predictive(self.model, posterior_samples=samples, return_sites=("_RETURN",'obs')).forward(x_test)
        
        #confidence intervall
        convidence_intervalls=posterior_predictive['_RETURN'].detach().cpu().numpy()
        y_pred=np.percentile(convidence_intervalls, percentiles, axis=0).T
        
        
        y_median_conv=y_pred[:,1].reshape(-1,1)
        y_lower_upper_quantil_conv=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
         
        #predictive intervall
        predictive_intervalls=posterior_predictive['obs'].detach().cpu().numpy()
        y_pred=np.percentile(predictive_intervalls, percentiles, axis=0).T
        
        y_median_pred=y_pred[:,1].reshape(-1,1)
        y_lower_upper_quantil_pred=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
         
        return(y_median_conv, y_lower_upper_quantil_conv, y_median_pred, y_lower_upper_quantil_pred)
        
# SVI Regression
class bay_reg_svi:
    def __init__(self,model, guide=None, epochs=500, lr=0.05, cuda=False):

        self.model=model
        self.cuda=cuda
        if guide != None:
            self.guide=guide
        else:
            self.guide=AutoDiagonalNormal(model)
        self.optimizer=optim.Adam({"lr": lr})
        
        self.svi=SVI(self.model,
                     self.guide,
                     self.optimizer,
                     loss=Trace_ELBO())
        
        self.epochs=epochs

    def fit(self, x_train, y_train, verbose=False):

        if self.cuda == True:
            x_train=torch.tensor(x_train).cuda()
            y_train=torch.tensor(y_train).cuda()   
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            x_train=torch.tensor(x_train)
            y_train=torch.tensor(y_train)

        for i in range(self.epochs):
            elbo = self.svi.step(x_train, y_train)
            if verbose:
                if i % 100 == 0:
                    print("Elbo loss: {}".format(elbo))
        return self
                    
    def predict(self, x_test, num_samples=500, percentiles=(5.0, 50.0, 95.0)):
        x_test=torch.tensor(x_test)
        posterior_predictive= Predictive(self.model, guide=self.guide, num_samples=800, return_sites=("_RETURN",'obs')).forward(x_test)
        
        #confidence intervall
        convidence_intervalls=posterior_predictive['_RETURN'].detach().cpu().numpy()
        
        y_pred=np.percentile(convidence_intervalls, percentiles, axis=0).T
        
        y_median_conv=y_pred[:,1].reshape(-1,1)
        y_lower_upper_quantil_conv=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
         
        #predictive intervall
        predictive_intervalls=posterior_predictive['obs'].detach().cpu().numpy()
        y_pred=np.percentile(predictive_intervalls, percentiles, axis=0).T
        
        y_median_pred=y_pred[:,1].reshape(-1,1)
        y_lower_upper_quantil_pred=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
         
        return(y_median_conv, y_lower_upper_quantil_conv, y_median_pred, y_lower_upper_quantil_pred)
    