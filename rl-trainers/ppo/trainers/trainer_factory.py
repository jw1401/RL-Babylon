TRAINER_REGISTRY = {}

def register_trainer(name):

    def deco(cls):
        TRAINER_REGISTRY[name.lower()] = cls
        return cls
    
    return deco

def make_trainer(name: str, **kwargs):

    name = name.lower()
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' nicht registriert.")
    
    cls = TRAINER_REGISTRY[name]
    instance = cls(**kwargs)

    return instance