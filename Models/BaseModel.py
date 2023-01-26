from abc import ABC, abstractmethod

'''
Unsure if I will be using abstract classes for every model I create.
Leaving it here just in case.
'''

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, Y):
        ...
    
    @abstractmethod
    def __repr__(self):
        '''
        Each model such have a special string representation to see the config
        '''
        ...