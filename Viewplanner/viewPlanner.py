from abc import ABC, abstractmethod

class NBVPlanner(ABC):
    def __init__(self, robot_sensor, partial_model):
        self.robot_sensor = robot_sensor
        self.partial_model = partial_model

    @abstractmethod
    def savePartialModel(self):
        pass
    
    @abstractmethod
    def updateWithScan(self):
        pass

    @abstractmethod
    def PlanNBV(self):
        # Code to plan the next best view
        pass


    