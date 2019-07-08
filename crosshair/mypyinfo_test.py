import unittest

from crosshair import mypyinfo


class MyPyInfoTest(unittest.TestCase):
    
    def test_info(self):
        #print(mypyinfo.__file__)
        info = mypyinfo.mypy_info([mypyinfo.__file__])
        self.assertTrue('crosshair.mypyinfo' in info.modules.keys())
        fninfo = info.lookup(mypyinfo.mypy_info)
        # first line creates a new Options object
        #print(fninfo)
        #print(fninfo.body)
        #print(fninfo.body.body)
        #print(fninfo.body.body[0])
        #print(fninfo.body.body[0].rvalue)
        #print(fninfo.body.body[0].rvalue.callee)
        self.assertEqual(fninfo.body.body[0].rvalue.callee.fullname,
              'mypy.options.Options')
    
    def Z_test_useless1(self):
        info = mypyinfo.mypy_info([mypyinfo.__file__])
        self.assertTrue('crosshair.mypyinfo' in info.modules.keys())
        fninfo = info.lookup(mypyinfo._useless1)
        print(fninfo.body.body[0])
        print(fninfo.body.body[0].expr)
        print(fninfo.body.body[0].expr.method_type)
        print(dir(fninfo.body.body[0].expr.method_type))
        print(fninfo.body.body[0].expr.method_type.arg_types)

    def Z_test_useless2(self):
        info = mypyinfo.mypy_info([mypyinfo.__file__])
        fninfo = info.lookup(mypyinfo._useless2)
        #print(fninfo.body.body[2])
        #print(fninfo.body.body[2].rvalue)
        #print(fninfo.body.body[2].rvalue.callee)
        self.assertEqual(fninfo.body.body[2].rvalue.callee.fullname,
                         'mypy.options.Options')
    

if __name__ == '__main__':
    unittest.main()
