import unittest
from mildew import Interpreter

class MildewUnittest(unittest.TestCase):
    def test1(self):
        interpreter = Interpreter()
        assert(interpreter.evaluate("1 + 5") == 6)
        assert(interpreter.evaluate("2 ** 3 ** 0") == 2)
        # TODO more unittests

if __name__ == "__main__":
    unittest.main()