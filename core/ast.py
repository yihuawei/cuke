import copy
import core

MIN_INT = -2147483648
MAX_INT = 2147483647

op_mapping = {'add':'+', 'sub':'-', 'mul':'*', 'floordiv':'/', 'truediv':'/'}
math_op = ['round', 'abs']
cmp_op = ['bigger', 'smaller']
func_op = ['index', 'apply', 'reduce', 'aggr', 'einsum', 'setval']


def is_int_var(v):
    return isinstance(v, Tensor) and v.dtype == 'int' and len(v.ref_size) == 0

def is_scalar(v):
    return isinstance(v, int|float) or (isinstance(v, Tensor) and len(v.ref_size) == 0)

def is_1dint_tensor(v):
    return isinstance(v, Tensor) and v.dtype == 'int' and len(v.ref_size) == 1


def eval_const_expr(e):
    if type(e) == TensorOp and (e.op_type in op_mapping):
        lhs = eval_const_expr(e.operators[0])
        if lhs != None:
            rhs = eval_const_expr(e.operators[1])
            if rhs != None:
                match e.op_type:
                    case 'add':
                        return lhs + rhs
                    case 'sub':
                        return lhs - rhs
                    case 'mul':
                        return lhs * rhs
                    case 'floordiv':
                        return lhs // rhs
                    case 'truediv':
                        return lhs / rhs
                    case _:
                        return None
            else:
                return None
        else:
            return None
    elif type(e) == Const:
        return e.val
    else:
        return None

def has_same_value(e1, e2):
    if type(e1) != type(e2):
        return False
    elif type(e1) == Var or type(e1) == Tensor:
        return e1.id == e2.id
    elif type(e1) == Const:
        if e1.dtype == 'int' and e2.dtype == 'int':
            return e1.val == e2.val
        elif e1.dtype == 'slice' and e2.dtype =='slice':
            return has_same_value(e1.val.start, e2.val.start) and has_same_value(e1.val.stop, e2.val.stop) and has_same_value(e1.val.step, e2.val.step)
        else:
            return False
    elif type(e1) == TensorOp:
        if e1.op_type != e2.op_type:
            return False
        elif e1.op_type in op_mapping:
            return has_same_value(e1.operators[0], e2.operators[0]) and has_same_value(e2.operators[1], e2.operators[1])
        else:
            if len(e1.operators) != len(e2.operators):
                return False
            else:
                for i in range(len(e1.operators)):
                    if not has_same_value(e1.operators[i], e2.operators[i]):
                        return False
    return True

def is_same_size(s1, s2):
    if(len(s1) != len(s2)):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if type(s1[i]) == type(s2[i]):
                return has_same_value(s1[i], s2[i])
            else:
                return False

    return True


def bigger(x, y):
    return TensorOp('bigger', x, y)

def smaller(x, y):
    return TensorOp('smaller', x, y)


class ASTNode:
    nuniq = 0
    def __init__(self):
        self.decl = []
        self.eval = None
        self.ref_count = 0
        self.id = ASTNode.nuniq
        ASTNode.nuniq += 1
        self.valid = True



class Tensor(ASTNode):
    def __init__(self, name, size:list|tuple, dtype='float', fix_size=[], is_arg=True):
        super().__init__()

        self.is_arg = is_arg
        self.name = name
        self.ref_size = []
        for s in size:
            if is_int_var(s):
                self.ref_size.append(s)
            elif type(s) == int:
                self.ref_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.fix_size = []
        for s in fix_size:
            if is_int_var(s):
                self.fix_size.append(s)
            elif type(s) == int:
                self.fix_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.dtype = dtype

    def __sub__(self, other):
        return TensorOp('sub', self, other)

    def __add__(self, other):
        return TensorOp('add', self, other)

    def __mul__(self, other):
        return TensorOp('mul', self, other)

    def __floordiv__(self, other):
        return TensorOp('floordiv', self, other)
	
    def __matmul__(self, other):
        return TensorOp('einsum', self, other, 'ij,jk->ik')

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice, Tensor)):
            return TensorOp('index', self, idx)
        else:
            raise TypeError('invalid index type')

    def apply(self, func, axis=0):
        if callable(func):
            from core.ast2ir import gen_ir
            op = TensorOp('apply', self, func, axis)
            return op
        else:
            raise TypeError('must apply a callable function')

    def reduce(self, func, init, axis=0):
        if callable(func) and callable(init):
            from core.ast2ir import  gen_ir
            op = TensorOp('reduce', self, func, init, axis)
            return op
        else:
            raise TypeError('reduce must use a callable function')

    def sum(self, axis=0):
        func = lambda x, y: x + y
        init = lambda x: x.setval(0)
        return self.reduce(func, init, axis)

    def max(self, axis=0):
        func = lambda x, y: bigger(x, y)
        init = lambda x: x.setval(MIN_INT)
        return self.reduce(func, init, axis)

    def min(self, axis=0):
        func = lambda x, y: smaller(x, y)
        init = lambda x: x.setval(MAX_INT)
        return self.reduce(func, init, axis)


    def aggr(self, func, init, indices, axis=0, size=None):
        if callable(func) and callable(init):
            from core.ast2ir import gen_ir
            op = TensorOp('aggr', self, func, init, indices, axis, size)
            gen_ir(op)
            return op
        else:
            raise TypeError('aggr must use a callable function')

    def aggr_sum(self, indices, axis=0, size=None):
        func = lambda x, y: x + y
        init = lambda x: x.setval(0)
        return self.aggr(func, init, indices, axis, size)

    def aggr_max(self, indices, axis=0, size=None):
        func = lambda x, y: bigger(x, y)
        init = lambda x: x.setval(MIN_INT)
        return self.aggr(func, init, indices, axis, size)

    def aggr_min(self, indices, axis=0, size=None):
        func = lambda x, y: smaller(x, y)
        init = lambda x: x.setval(MAX_INT)
        return self.aggr(func, init, indices, axis, size)



    def setval(self, val):
        return TensorOp('setval', self, val)

    def _size(self):
        return self.fix_size + self.ref_size

    def size(self):
        s = self._size()  # s is a list of int, Var, or Const
        if len(s) > 1:
            return Tensor(f'{self.name}_size', val=s, dtype='int')
        elif len(s) == 1:
            if type(s[0]) == int:
                return Const(s, dtype='int')
            else:
                return s[0]
        else:
            return Const(0, dtype='int')



    def round(self):
        return TensorOp('round', self)

    def abs(self):
        return TensorOp('abs', self)

    def _gen_ir(self):
        return core.ast2ir.gen_ir(self)


class Var(Tensor):
    def __init__(self, name, dtype='int', is_arg=True):
        super().__init__(name, [], dtype, [], is_arg)



# const is var without name
class Const(Var):
    nconsts = 0
    def __init__(self, val, dtype):
        super().__init__(f'c{Const.nconsts}', dtype)
        Const.nconsts += 1
        # slice is considered constant because once the slice is created its start, stop, step cannot be reassigned
        # however, start, stop, step themselves can be variables
        if dtype == 'slice':
            assert type(val.start) == int or is_int_var(val.start)
            assert type(val.stop) == int or is_int_var(val.stop)
            assert type(val.step) == int or is_int_var(val.step)
        self.val = val


def einsum(exp: str, tensor1, tensor2):
    return TensorOp('einsum', tensor1, tensor2, exp)

class TensorOp(Tensor):
    Types = func_op + list(op_mapping.keys()) + math_op + cmp_op

    def __init__(self, op_type, *operators):
        assert op_type in TensorOp.Types
        self.compute = []
        self.compute_block = []
        self.output_order = []

         # TODO: infer result data type
        dtype = operators[0].dtype
        self.operators = []
        for opr in operators:
            # an index can be referenced multiple times in the ast, we should create duplicate copies so that they can bind with different loop iterates
            if type(opr) == TensorOp and opr.op_type == 'index' and opr.ref_count >= 1:
                new_opr = copy.copy(opr)
                new_opr.ref_count = 1
                new_opr.eval = None
                new_opr.decl.clear()
                new_opr.compute.clear()
                self.operators.append(new_opr)
            else:
                self.operators.append(opr)
                if isinstance(opr, ASTNode):
                    opr.ref_count += 1

        if op_type in op_mapping or op_type in cmp_op:

            if type(self.operators[0]) == int:
                self.operators[0] = Const(self.operators[0], 'int')
            elif type(operators[0]) == float:
                self.operators[0] = Const(self.operators[0], 'float')
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            elif type(operators[1]) == float:
                self.operators[1] = Const(self.operators[1], 'float')
            assert is_same_size(self.operators[0]._size(), self.operators[1]._size()) or len(self.operators[0]._size()) == 0 or len(self.operators[1]._size()) == 0
            if len(self.operators[0]._size()) < len(self.operators[1]._size()):
                self.operators[0], self.operators[1] = self.operators[1], self.operators[0]

            ref_size = self.operators[0].fix_size + self.operators[0].ref_size
            fix_size = []

        elif op_type == 'einsum':
            exp = self.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            op1_size = self.operators[0]._size()
            op2_size = self.operators[1]._size()
            ref_size = []
            fix_size = []
            for i in output:
                pos1 = input1.find(i)
                if pos1 >= 0:
                    ref_size.append(op1_size[pos1])
                else:
                    pos2 = input2.find(i)
                    if pos2 >= 0:
                        ref_size.append(op2_size[pos2])
                    else:
                        raise IndexError('index not found!')

        elif op_type == 'index':
            ref_size = self.operators[0].ref_size[1:]
            fix_size = self.operators[0].fix_size[:]
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            elif type(self.operators[1]) == slice:
                start = self.operators[1].start
                if start == None:
                    start = Const(0, 'int')
                elif type(start) == int:
                    start = Const(start, 'int')
                stop = self.operators[1].stop
                if stop == None:
                    stop = self.operators[0].ref_size[0]
                elif type(stop) == int:
                    stop = Const(stop, 'int')
                step = self.operators[1].step
                if step == None:
                    step = Const(1, 'int')
                elif type(step) == int:
                    step = Const(step, 'int')

                self.operators[1] = Const(slice(start, stop, step), 'slice')
                csize = eval_const_expr((stop - start)//step)
                if csize != None:
                    fix_size.append(csize)
                else:
                    if step.val == 1:
                        fix_size.append(stop-start)
                    else:
                        fix_size.append((stop - start)//step)
            elif is_int_var(self.operators[1]):
                self.operators[1] = self.operators[1]
            elif is_1dint_tensor(self.operators[1]):
                fix_size.append(self.operators[1]._size()[0])
            else:
                raise TypeError('index must be int, Var of int, or 1d int Tensor')

        elif op_type == 'apply':
            assert type(self.operators[2]) == int
            axis = self.operators[2]
            self.operators[2] = Const(axis, 'int')

            data_size = self.operators[0]._size()
            item_size = data_size[:axis] + data_size[axis + 1:]
            if (len(item_size) > 0):
                item = Tensor(f'item_of_{self.operators[0].name}', item_size,
                              self.operators[0].dtype, [], False)
            else:
                item = Var(f'item_of_{self.operators[0].name}', self.operators[0].dtype, False)

            ret = self.operators[1](item)
            dtype = ret.dtype
            ref_size = [self.operators[0]._size()[axis]] + ret._size()
            fix_size = []
            self.operators.append(item)
            self.operators.append(ret)

        elif op_type == 'reduce':
            assert type(self.operators[3]) == int
            axis = self.operators[3]
            self.operators[3] = Const(axis, 'int')
            ref_size = self.operators[0]._size()[:axis] + self.operators[0]._size()[axis+1:]
            fix_size = []
            dtype = self.operators[0].dtype
            if (len(ref_size) > 0):
                item1 = Tensor(f'item1_of_{self.operators[0].name}', ref_size,
                               self.operators[0].dtype, [], False)
                item2 = Tensor(f'item2_of_{self.operators[0].name}', ref_size,
                               self.operators[0].dtype, [], False)
            else:
                item1 = Var(f'item1_of_{self.operators[0].name}', self.operators[0].dtype, False)
                item2 = Var(f'item2_of_{self.operators[0].name}', self.operators[0].dtype, False)

            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type == 'aggr':
            assert is_1dint_tensor(self.operators[3])
            assert type(self.operators[4]) == int
            axis = self.operators[4]
            self.operators[4] = Const(axis, 'int')
            if self.operators[5] == None:
                self.operators[5] = self.operators[3].ref_size[0]
            else:
                assert is_int_var(self.operators[5])
                if type(self.operators[5]) == int:
                    self.operators[5] = Const(self.operators[5], 'int')
            ref_size = [self.operators[5]] + self.operators[0]._size()[:axis] + self.operators[0]._size()[axis+1:]
            fix_size = []
            dtype = self.operators[0].dtype
            if (len(ref_size) > 1):
                item1 = Tensor(f'item1_of_{self.operators[0].name}', ref_size[1:],
                               self.operators[0].dtype, [], False)
                item2 = Tensor(f'item2_of_{self.operators[0].name}', ref_size[1:],
                               self.operators[0].dtype, [], False)
            else:
                item1 = Var(f'item1_of_{self.operators[0].name}', self.operators[0].dtype, False)
                item2 = Var(f'item2_of_{self.operators[0].name}', self.operators[0].dtype, False)
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type in math_op:
            ref_size = self.operators[0].ref_size
            fix_size = []
            if op_type == 'round':
                dtype = 'int'
            elif op_type == 'abs':
                dtype = self.operators[0].dtype

        elif op_type == 'setval':
            ref_size = self.operators[0].ref_size
            fix_size = self.operators[0].fix_size
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            elif type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'float')


        name = f'{op_type}_' + '_'.join([op.name if hasattr(op, 'name') else '' for op in self.operators])

        super().__init__(name, ref_size, dtype, fix_size)

        self.op_type = op_type

        # call the init function for reduce and aggr
        if self.op_type in ('reduce', 'aggr'):
            self.operators[2] = self.operators[2](self)






