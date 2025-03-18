

class ViewPlanner:

    def __init__(self, method):
        self.method = self.__loadmethod()

    def __loadmethod(self):
        #here we will load the corresponding method
        pass


    def forward(self, i, direccion):
        
        """
        we recieve direction to the data then the method access to the requiered data
        and return a prediction

        How to do it for multiple returned parameters f.e. [x,y,z] o [R|t] o class?

        """
        nbv = self.method.nbv()
        return nbv
