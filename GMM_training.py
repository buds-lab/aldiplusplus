import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class GMMTraining():

    def __init__(self, values):
        self.values = np.array([[val] for val in values])
        self.x_values = np.linspace(0, 1, 1000)

        '''
        p_values = np.array([ [val] for val in df_pD_values.p.values])
        D_values = np.array([ [val] for val in df_pD_values.D.values])
    
        N = np.arange(1, 11)
        models = [None for i in range(len(N))]
        for i in range(len(N)):
            models[i] = GaussianMixture(N[i]).fit(p_values)
        
        AIC = [m.aic(p_values) for m in models]
        BIC = [m.bic(p_values) for m in models]
        best_GMM = models[np.argmin(AIC)]

        logprob = best_GMM.score_samples(x_values.reshape(-1, 1))
        responsibilities = best_GMM.predict_proba(x_values.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        '''

        N = np.arange(1, 11)
        self.models = [None for i in range(len(N))]
        for i in range(len(N)):
            self.models[i] = GaussianMixture(N[i]).fit(self.values)
        
        self.AIC = [m.aic(self.values) for m in self.models]
        self.BIC = [m.bic(self.values) for m in self.models]
        self.best_GMM = self.models[np.argmin(self.AIC)]

        self.logprob = self.best_GMM.score_samples(self.x_values.reshape(-1, 1))
        self.responsibilities = self.best_GMM.predict_proba(self.x_values.reshape(-1, 1))
        self.pdf = np.exp(self.logprob)
        self.pdf_individual = self.responsibilities * self.pdf[:, np.newaxis]

    def get_pdf(self):
        return self.pdf

    def get_individual_pdf(self):
        return self.pdf_individual

    def get_x_values(self):
        return self.x_values
