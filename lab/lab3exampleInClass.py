def get_obj(ir: (Index, Scalar)):
    obj = ir
    while hasattr(obj, 'dobject'):
        obj = obj.dobject
    return obj


def replace_index_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            replace_index_with_scalar(l, old, new)
    elif type(ir) == Loop:
        replace_index_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Index, Scalar):
            obj = get_obj(ir.left)
            if obj == old:
                ir.left = new
        else:
            replace_index_with_scalar(ir.left, old, new)
        
        if type(ir.right) in (Index, Scalar):
            obj = get_obj(ir.right)
            if obj == old:
                ir.right = new
        else:
            replace_index_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Index, Scalar):
            obj = get_obj(ir.lhs)
            if obj == old:
                ir.lhs = new
        else:
            replace_index_with_scalar(ir.lhs, old, new)
        
        if type(ir.rhs) in (Index, Scalar):
            obj = get_obj(ir.rhs)
            if obj == old:
                ir.rhs = new
        else:
            replace_index_with_scalar(ir.rhs, old, new)


def rebind_iterate(ir, old_idx, new_idx):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            rebind_iterate(l, old_idx, new_idx)
    elif type(ir) == Assignment:
        rebind_iterate(ir.left,  old_idx, new_idx)
        rebind_iterate(ir.right, old_idx, new_idx)
    elif type(ir) == Index:
        if ir.index == old_idx:
            ir.index = new_idx:
    #Some other types 


def fusable_level(node0, node1):
    def _fusable_level(loop0, loop1, level):
        if  type(loop0)!=Loop or type(loop1)!=Loop:
            return level
        if loop0.start==loop1.start and loop0.end==loop1.end and loop0.step==loop1.step:
            return _fusable_level(loop0.body[0], loop1.body[0], level+1)
        else:
            return level
    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    return _fusable_level(loop0, loop1, 0)


def move_ir(node0, node1, move_level):
    def _move_ir(loop0, loop1, cur_level):
        if cur_level==move_level-1:
            loop1.body = loop0.body + loop1.body 
        else:
            _move_ir(loop0.body[0], loop1.body[0], cur_level+1) 

    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    _move_ir(loop0, loop1, 0)


def get_rhs(node):
    def _get_rhs(ir):
        if type(ir)==Loop:
            return _get_rhs(ir.body[0])
        else:
            print(ir)
            assert type(ir)==Assignment
            return ir.rhs
    return _get_rhs(node.compute[0])

def rebind_iterate():
    pass



            body_level = fusable_level(node.operators[0], node)
            if body_level>0:
                # move_ir(node.operators[0], node, body_level)
                replace_index_with_scalar(node.compute[0], node.operators[0].eval, get_rhs(node.operators[0]))
        if type(node) == TensorOp and node.op_type in elementwise_op and type(node.operators[0]) == TensorOp and node.operators[0].op_type in elementwise_op:
