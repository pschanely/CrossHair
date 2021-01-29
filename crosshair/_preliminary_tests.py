import os
import unittest

class PreliminariesTest(unittest.TestCase):

    def test_PYTHONHASHSEED_is_zero(self) -> None:
        self.assertEqual(
            os.getenv('PYTHONHASHSEED'),
            '0',
            'CrossHair tests should be run with the PYTHONHASHSEED '
            'environement variable set to 0. Some other tests rely on this '
            'for deterministic behavior.'
        )

if __name__ == '__main__':
    unittest.main()
