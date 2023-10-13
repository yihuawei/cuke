from core.ast import *
import cset
from cset.ast2ir import *

class Set(ASTNode):
    capacity = 20000

    def __init__(self, storage):
        super().__init__()
        if isinstance(storage, Tensor):
            self.storage = storage
            # print(type(storage))
        else:
            raise TypeError("init val of set must be a tensor")

        self.name = f'set_{self.storage.name}'
        self.nelem = Var(f'{self.name}_nelem', 'int', False)
        if type(self) == Set:
            self.nelem = Const(1, 'int')
            for s in self.storage._size():
                self.nelem = self.nelem * s
        else:
            self.nelem = Var(f'{self.name}_nelem', 'int', False)
        
        # self.nelem = self.storage._size()
        self.dtype = storage.dtype
        self.is_arg = False
    
    def _size(self):
        return self.storage._size()
    def _gen_ir(self):
        return cset.ast2ir.gen_ir(self)

    def __add__(self, other):
        return SetOp('add', self, other)

    def apply(self, func, axis=0):
        if callable(func):
            op = SetOp('apply', self, func, axis)
            return op
        else:
            raise TypeError('must apply a callable function')
    
    def filter(self, condition):
        if callable(condition):
            op = SetOp('filter', self, condition)
            return op
        else:
            raise TypeError('must apply a callable function')
    
    def num_elem(self):
        op = SetOp('nelem', self)
        return op
    def sum(self):
        op = SetOp('sum', self)
        return op
    
    # def __getitem__(self, idx):
    #     if isinstance(idx, (int, slice, Tensor, Set)):
    #         return SetOp('index', self, idx)
    #     else:
    #         raise TypeError('invalid index type')

    # def intersect(self, other):
    #     assert type(other) == Set
    #     return SetOp('intersect', self, other)

# class IsIn:
# 	def __init__(self, B:Set):
	
# 	#A中的Item
# 	def __call__(item):
# 		return SetOp('Search', item, B)

class BinarySearch():
    def __init__(self, target_set):
        self.target_set = target_set
    def __call__(self, item):
        return SetOp('search', self.target_set, item)

class SetOp(Set):
    Types = ['intersect', 'apply']

    def __init__(self, op_type, *operators):
        
        self.operators = list(operators)
        self.op_type = op_type

        if op_type == 'apply':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            input_storage_size = input_storage._size()
            input_storage_dtype = input_storage.dtype
            
            # assert len(input_storage_size)==1
            axis =  self.operators[2]
            self.operators[2] = Const(axis, 'int')
            data_size = input_storage_size
            item_size = data_size[:axis] + data_size[axis + 1:]
            if (len(item_size) > 0):
                item = Tensor(f'item_of_{input_storage_name}', item_size,
                              input_storage_dtype, [], False)
            else:
                item = Var(f'item_of_{input_storage_name}', input_storage_dtype, False)
            self.operators.append(item)

            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Tensor(res_storage_name, [self.capacity], dtype = 'int', is_arg=False))
        
        elif op_type == 'filter':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            input_storage_size = input_storage._size()
            input_storage_dtype = input_storage.dtype
            assert len(input_storage_size)==1
            assert callable(self.operators[1])

            item = Var(f'item_of_{input_storage_name}', input_storage_dtype, False)
            self.operators.append(item)
            # cond_set_op = self.operators[1](item)
            # self.operators[1] =  cond_set_op   
           

            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Tensor(res_storage_name, [self.capacity], dtype = 'int', is_arg=False))
        
        elif op_type == 'search':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'res_of_{res_storage_name}', 'int', False))

        elif op_type == 'nelem':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'nelem_of_{res_storage_name}', 'int', False))
        
        elif op_type == 'sum':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'sum_of_{res_storage_name}', 'int', False))

        
