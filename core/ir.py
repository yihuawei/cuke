class IR:
    def __init__(self):
        # astnode tracks the location of this IR in the AST
        self.astnode = None


class Code(IR):
    def __init__(self, code, keywords: dict):
        self.code = code
        self.keywords = keywords


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
        self.size = self.left.size



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
    def __init__(self, dtype: str, name: str = None, is_arg = False):
        super().__init__(dtype, [])
        self.__name__ = name if name else f's{self.dobject_id}'
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
    def __init__(self, dtype: str, size: tuple, name: str = None, is_arg = False):
        super().__init__(dtype, size)
        self.__name__ = name if name else f'arr{self.dobject_id}'
        self.is_arg = is_arg

    def __getitem__(self, item):
        return f'{self.__name__}[{item}]'

    def name(self):
        return self.__name__


class Math(IR):
    def __init__(self, val, type):
        self.val = val
        self.type = type

class Indexing(DOject):
    def __init__(self, dobject, idx):
        assert dobject != None and type(dobject) in (Slice, Ndarray, Indexing)
        assert idx != None and type(idx) in (Scalar, Literal, Indexing, Expr)
        self.dobject = dobject
        self.idx = idx

        if type(self.dobject) in (Ndarray, Slice):
            if type(idx) == Literal and idx.val == -1:
                # idx is unspecified, which means the Indexing is a range of indice stored in dobject, so the size of Indexing should the same as the dobject
                size = dobject.size[:]
                self.ref_point = 1
            else:
                # idx is a specific Scalar, Literal, or Indexing, in any case, the size of the Indexing operation should be as follows
                # ref_point should be the next dimension if the node is further Indexed
                size = idx.size + dobject.size[1:]
                self.ref_point = len(idx.size)
        else:
            # dobject is an Indexing
            size = dobject.size[:dobject.ref_point] + idx.size + dobject.size[dobject.ref_point+1:]
            self.ref_point = dobject.ref_point + len(idx.size)

        super().__init__(dobject.dtype, size)


class Decl(IR):
    def __init__(self, dobject: (Scalar, Ndarray)):
        super().__init__()
        self.dobject = dobject