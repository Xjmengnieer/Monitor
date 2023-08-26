from Registry import registry

CLASSIFIER = registry('classifier')

def bulid_classifier(cfg):
    return CLASSIFIER.build_model(cfg)