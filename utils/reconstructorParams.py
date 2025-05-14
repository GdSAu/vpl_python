import yaml

class Params:

    def __init__(self,filename):
        try: 
            with open(filename, 'r') as file:
                self.params = yaml.safe_load(file)
                self.__extractParameters()
        except FileNotFoundError:
            print(f"Error: File not found'{filename}'")
        except yaml.YAMLError as e:
            print(f"Error while parsing file: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def __extractParameters(self):
        if not self.params:
            print(f"Warning: YAML file is empty or could not be loaded.")
            self.result = {}
        else:    
            self.result = self.params

    def getParameter(self, path: str):
        
        if not self.result:
            # Si aún no se ha llamado a extractParameters, hacerlo ahora
            self.extractParameters()
            
        parts = path.split('.')
        current = self.result
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
                
        return current