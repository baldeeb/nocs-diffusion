class ModelReturnType:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
    def __getitem__(self, i):
        return list(self.__dict__.values())[i]
    