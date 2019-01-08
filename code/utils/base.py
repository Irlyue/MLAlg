class BaseModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def __repr__(self):
        values = ', '.join('{}={}'.format(k, v) for k, v in self.__dict__.items())
        return '{}({})'.format(type(self).__name__, values)

    def on_end_fit(self):
        raise NotImplementedError
