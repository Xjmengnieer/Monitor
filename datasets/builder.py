from Registry import registry

Datasets = registry('Data')

def bulid_dataset(cfg, train = True):
    return Datasets.build_dataset(cfg, train)