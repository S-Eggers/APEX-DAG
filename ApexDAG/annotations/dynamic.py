import typing

class Dynamic:
    """
    A custom type annotation for static analysis to flag variables
    that will be subject to dynamic dispatch at runtime.

    For standard type checkers, this will behave like a Union.
    """
    def __class_getitem__(cls, item: typing.Any) -> typing.Union:
        """Makes the class subscriptable, e.g., Dynamic[Dog, Cat]."""
        return typing.Union[item]