import reef_models as rm

#********************************************************************************************

class DicoModels():
    
    def __init__(self):
        
        self.models = {
            'reef'          : rm.reef,
            'reef_platform' : rm.reef_platform,
            'reef_eros'     : rm.reef_eros
        }