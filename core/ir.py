class IR:
    def __init__(self):
        # astnode tracks the location of this IR in the AST
        self.astnode = None

class DOject(IR):
    nobjects = 0

    def __init__(self, dtype: str, size: (list, tuple)):
        super().__init__()
        self.dobject_id = DOject.nobjects
        DOject.nobjects += 1
        self.dtype = dtype
        self.size = size


class Expr(IR):
    def __init__(self, left, right, op: str):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op



class Assignment(IR):
    def __init__(self, lhs, rhs, op=None):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op



class Loop(IR):
    loop_id = 0

    def __init__(self, start, end, step, body: list):
        super().__init__()
        self.lid = Loop.loop_id
        Loop.loop_id += 1
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        self.iterate = Scalar('int', f'_l{self.lid}')


class Scalar(DOject):
    def __init__(self, dtype: str, name: str = None, is_arg = False, val = None):
        super().__init__(dtype, [])
        self.__name__ = name if name else f's{self.dobject_id}'
        self.val = val
        self.is_arg = is_arg
    def name(self):
        return self.__name__


class Literal(DOject):
    def __init__(self, val: (int, float), dtype: str):
        super().__init__(dtype, [])
        self.val = val


class Slice(IR):
    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = 'int'
        self.size = [Expr(Expr(self.stop, self.start, '-'), self.step, '/')]


class Ndarray(DOject):
    def __init__(self, dtype: str, size: tuple, name: str = None, is_arg = False, val = None):
        super().__init__(dtype, size)
        self.__name__ = name if name else f'arr{self.dobject_id}'
        self.val = val # val is None, 0, or 1
        self.is_arg = is_arg

    def __getitem__(self, item):
        return f'{self.__name__}[{item}]'

    def name(self):
        return self.__name__


class Indexing(DOject):
    def __init__(self, dobject, idx):
        assert dobject != None and type(dobject) in (Slice, Ndarray, Indexing)
        assert idx != None and type(idx) in (Scalar, Literal, Indexing)
        self.dobject = dobject
        self.idx = idx
        # TODO: infer index sizes
        if type(self.dobject) in (Ndarray, Slice):
            size = idx.size + dobject.size[1:]
            self.ref_point = len(idx.size)
        else:
            size = dobject.size[:dobject.ref_point] + idx.size + dobject.size[dobject.ref_point+1:]
            self.ref_point = dobject.ref_point + len(idx.size)

        super().__init__(dobject.dtype, size)






class Decl(IR):
    def __init__(self, dobject: (Scalar, Ndarray)):
        super().__init__()
        self.dobject = dobject