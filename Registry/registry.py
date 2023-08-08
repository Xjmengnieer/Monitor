from inspect import isfunction

class registry():
    def __init__(self, name: str):
        self.model_dict = {}
        self.func_dict = {}
        self.name = name
    
    def register_module(self, name: str, module = None):
        if module is not None:
            self.model_dict[name] = module
            return module
        
        def _registty(obj):
            obj_name = obj.__name__
            if isfunction(obj):
                self.func_dict[obj_name] = obj
            else:
                self.model_dict[obj_name] = obj

            return obj

        return _registty

    def build(self, cfg):
        if cfg.model.type in self.func_dict:
            obj = self.func_dict[cfg.model.type](**cfg.model)
        else:
            obj =  self.model_dict[cfg.model.type](**cfg.model)
        
        return obj