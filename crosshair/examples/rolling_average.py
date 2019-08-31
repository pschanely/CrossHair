from typing import List

class AverageableStack:
    '''
    A stack of numbers with a O(1) average() operation.
    
    inv: self.total == sum(self.values)
    '''
    values: List[int]
    total: int
    def __init__(self):
        self.values = []
        self.total = 0
    def push(self, val: int):
        '''post[self]: True'''
        self.values.append(val)
        self.total += val
    def pop(self) -> int:
        '''
        pre: self.values
        post[self]: True
        '''
        val = self.values.pop()
        self.total -= val
        return val
    def average(self):
        if not self.values:
            return 0
        return sum(self.values) / len(self.values)
    
       
