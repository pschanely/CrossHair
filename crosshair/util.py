import sys
import traceback


def debug(*a):
    return
    stack = traceback.extract_stack()
    frame = stack[-2]
    indent = len(stack) - 3
    print('{}{}() {}'.format(
        ' ' * indent, frame.name, ' '.join(map(str, a))), file=sys.stderr)


def walk_qualname(obj: object, name: str) -> object:
    '''
    >>> walk_qualname(list, 'append') == list.append
    True
    >>> class Foo:
    ...   class Bar:
    ...     def doit():
    ...       pass
    >>> walk_qualname(Foo, 'Bar.doit') == Foo.Bar.doit
    True
    '''
    for part in name.split('.'):
        if part == '<locals>':
            raise ValueError(
                'object defined inline are non-addressable(' + name + ')')
        if not hasattr(obj, part):
            raise Exception('Name "' + part + '" not found')
        obj = getattr(obj, part)
    return obj
