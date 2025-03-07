class InsufficientNegativeEdgesException(Exception):
    def __init__(self, message="The graph does not have enough negative edges"):
        self.message = message
        super().__init__(self.message)
