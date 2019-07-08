import unittest

from enforce import *


def foo(x:int) -> int:
    '''
    pre: 0 <= x <= 100
    post: return > x
    '''
    return x * 2

class CoreTest(unittest.TestCase):

    def test_enforce_and_unenforce(self) -> None:
        env = {'foo':foo, 'bar':lambda x:x, 'baz':42}
        backup = env.copy()
        with EnforcedConditions(env, interceptor=lambda f:(lambda x:x*3)):
            self.assertIs(env['bar'], backup['bar'])
            self.assertIs(env['baz'], 42)
            self.assertIsNot(env['foo'], backup['foo'])
            self.assertEqual(env['foo'](50), 150) # type:ignore
        self.assertIs(env['foo'], backup['foo'])
        
    def test_enforce_conditions(self) -> None:
        env = {'foo':foo}
        self.assertEqual(foo(-1), -2) # unchecked
        with EnforcedConditions(env):
            self.assertEqual(env['foo'](50), 100)
            with self.assertRaises(PreconditionFailed):
                env['foo'](-1)
            with self.assertRaises(PostconditionFailed):
                env['foo'](0)
                
            
if __name__ == '__main__':
    unittest.main()
    
