import ext.batch
from core.ast import *

class Batch(ASTNode):
    def __init__(self, base, bsize):
        super().__init__()
        self.base = base
        self.batch_size = bsize

    def __getattr__(self, item):
        if item == 'dtype':
            return self.base.dtype
        elif item == 'name':
            return self.base.name

    def _gen_ir(self):
        return ext.batch.ast2ir.gen_ir(self)




class BVec(Batch):
    def __init__(self, base):
        size = base._size()
        assert len(size) == 2
        self.dim = size[1]
        super().__init__(base, size[0])


    def __sub__(self, other):
        return BVec(self.base - other.base)

    def __add__(self, other):
        return BVec(self.base + other.base)

    def __mul__(self, other):
        return BVec(self.base + other.base)

    def __floordiv__(self, other):
        return BVec(self.base // other.base)




class BMat(Batch):
    def __init__(self, base):
        size = base._size()
        assert len(size) == 3
        self.dim1 = size[1]
        self.dim2 = size[2]
        super().__init__(base, size[0])




class BVar(Batch):
    def __init__(self, base):
        super().__init__()
        size = base._size()
        assert len(size) == 1
        self.base = base
        self.batch_size = size[0]


def bvv(v1: BVec, v2: BVec):
    assert v1.batch_size == v2.batch_size
    assert v1.dim == v2.dim

    return BatchOp('vec_mul_vec', v1, v2)

class BatchOp(Batch):
    Types = ['scal_mul_vec', 'vec_mul_vec', 'vec_mul_mat']

    def __init__(self, op_type, *operators):
        assert op_type in BatchOp.Types
        # TODO: infer result data type
        dtype = operators[0].dtype
        self.operators = list(operators)

        if op_type == 'vec_mul_vec':
            name = f'{op_type}_' + '_'.join([op.name if hasattr(op, 'name') else '' for op in self.operators])
            size = operators[0]._size()
            res = Tensor(name, (size[0], ), dtype)
            super().__init__(res, size[0])

        else: # TODO: complete other ops
            pass

        self.op_type = op_type

