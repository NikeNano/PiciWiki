from abc import ABC, abstractmethod
 
class AbstracExtractor(ABC):
 
    @abstractmethod
    def loadAndPreprocess(self):
        pass

    @abstractmethod
    def predict(self):
        pass