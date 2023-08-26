from inspect import isfunction

class registry():
    def __init__(self, name: str):
        self.model_dict = {}
        self.func_dict = {}
        self.name = name

        self.dataset_dict = {}
    
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

    def register_dataset(self, name: str, dataset = None):
        if dataset is not None:
            self.dataset_dict[name] = dataset
            return dataset
        
        def _registty(obj):
            obj_name = obj.__name__
            self.dataset_dict[obj_name] = obj

            return obj

        return _registty

    def build_model(self, cfg):
        if cfg.model.type in self.func_dict:
            obj = self.func_dict[cfg.model.type](**cfg.model)
        else:
            obj =  self.model_dict[cfg.model.type](**cfg.model)
        
        return obj

    def build_dataset(self, cfg, train = True):

        if train:
            obj = self.dataset_dict[cfg.datasets.type](**cfg.datasets.train)
        else:
            obj = self.dataset_dict[cfg.datasets.type](**cfg.datasets.val)
        
        return obj

            