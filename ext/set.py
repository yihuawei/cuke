from core.ast import *
class Set(ASTNode):
    def __init__(self, storage):
        super().__init__()
        if isinstance(storage, Tensor):
            self.storage = storage
        else:
            raise TypeError("init val of set must be a tensor")

        self.name = f'set_{self.storage.name}'

        t = Const(1, 'int')
        for s in self.storage._size():
            t = t * s
        self.nelem = Var(f'{self.name}_nelem', 'int', t)

    def num_elem(self):
        return self.nelem

    def intersect(self, other):
        assert type(other) == Set
        return SetOp('intersect', self, other)


class SetOp(Set):
    Types = ['intersect', 'difference']

    def __init__(self, operators):
        pass