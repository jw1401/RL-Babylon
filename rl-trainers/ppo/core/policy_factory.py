POLICY_REGISTRY = {}

def register_policy(name):
    def deco(cls):
        POLICY_REGISTRY[name.lower()] = cls
        return cls
    return deco


def make_policy(name: str, **kwargs):
    
    name = name.lower()
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Policy '{name}' nicht registriert.")
    cls = POLICY_REGISTRY[name]

    # filter kwargs to only those accepted by the policy
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return cls(**valid_kwargs)
