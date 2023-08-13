import pytest

from ml3m._docstring import format_docstring

#######################################################################################
#                                                                                     #
#                                  TESTS START HERE                                   #
#                                                                                     #
#######################################################################################


@pytest.mark.parametrize("kwd", [1, "hello", list(range(5))])
def test_format_docstring(kwd):
    @format_docstring(kwd=kwd)
    def func():
        """This is a function with {kwd}."""

    # Test formatting of function docstring
    assert func.__doc__ == f"This is a function with {kwd}."

    @format_docstring(kwd=kwd)
    class Klass:
        """This is a class with {kwd}."""

        @format_docstring(kwd=kwd)
        def meth(self):
            """This is a method with {kwd}."""

    # Test formatting of class docstring
    assert Klass.__doc__ == f"This is a class with {kwd}."

    # Test formatting of method docstring
    obj = Klass()
    assert obj.meth.__doc__ == f"This is a method with {kwd}."
