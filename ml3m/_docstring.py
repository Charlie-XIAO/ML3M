# This file is for docstring-relevant functionalities


def format_docstring(**kwargs):
    """Decorator that formats the docstring with specified keywords.

    Notes
    -----
    If the docstring is to be formatted and it includes curly braces (for example, when
    typing literals), one must use double curly braces instead.
    """
    def wrapped(func_or_cls):
        func_or_cls.__doc__ = func_or_cls.__doc__.format(**kwargs)
        return func_or_cls
    return wrapped
