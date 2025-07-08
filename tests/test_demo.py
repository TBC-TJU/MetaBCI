from .base_tmpl import BaseTmpl


def add(a, b):
    return a + b


def minus(a, b):
    return a - b


class TestDemo(BaseTmpl):

    def test_add(self):
        self.dbgPrint("3 == 1 + 2")
        self.assertEqual(3, add(1, 2))
        self.assertNotEqual(3, add(2, 2))

    def test_minus(self):
        self.assertEqual(1, minus(3, 2))
