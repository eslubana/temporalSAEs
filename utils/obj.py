class DictToObj:
    """
    A class to convert nested dictionaries to objects (to handle custom config objects).
    """
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = DictToObj(value)
        self.__dict__.update(d)
    
    def __getitem__(self, key):
        if key in self.__dict__.keys() or not isinstance(key, int):
            return self.__dict__[key]
        return list(self.__dict__)[key]