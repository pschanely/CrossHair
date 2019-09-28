import unittest

from crosshair.enforce import *


def foo(x:int) -> int:
    '''
    pre: 0 <= x <= 100
    post: _ > x
    '''
    return x * 2

class Pokeable:
    '''
    inv: self.x >= 0
    '''
    x :int = 1
    def poke(self) -> None:
        self.x += 1
    def pokeby(self, amount:int) -> None:
        '''
        pre: amount >= 0
        '''
        self.x += amount

        
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
                
    def test_class_enforce(self) -> None:
        env = {'Pokeable':Pokeable}
        old_id = id(Pokeable.poke)
        Pokeable().pokeby(-1)  # no exception (yet!)
        with EnforcedConditions(env):
            self.assertNotEqual(id(env['Pokeable'].poke), old_id)
            Pokeable().poke()
            with self.assertRaises(PreconditionFailed):
                Pokeable().pokeby(-1)
        self.assertEqual(id(env['Pokeable'].poke), old_id)
        
            
if __name__ == '__main__':
    unittest.main()
    
