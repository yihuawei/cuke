from core.ast import *
from core.ir import *
from opt.loop import *
import helpers



def get_first_unbind(index: (Indexing, Ndarray, Slice)):
    if type(index) == Indexing:
        x = get_first_unbind(index.dobject)
        if x != None:
            return x
        else:
            if type(index.idx) == Literal and index.idx.val == -1:
                return index
            else:
                y = get_first_unbind(index.idx)
                return y
    return None


def bind(index: (Indexing, Ndarray, Slice), idx):
    x = get_first_unbind(index)
    if x == None:
        return Indexing(index, idx)
    else:
        x.idx = idx
        return index


def replace_output(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            replace_output(l, old, new)
    elif type(ir) == Loop:
        replace_output(ir.body, old, new)
    elif type(ir) == Assignment:
        if ir.lhs == old:
            ir.lhs = new
        else:
            replace_output(ir.lhs, old, new)
    elif type(ir) == Indexing:
        if ir.dobject == old:
            ir.dobject = new
        else:
            replace_output(ir.dobject, old, new)

def gen_ir(node):
    assert isinstance(node, ASTNode)
    if len(node.compute) > 0 or len(node.decl) > 0 or node.eval:
        return node
    if type(node) == Const:
        if node.dtype != 'slice':
            assert type(node.val) == int or type(node.val) == float
            node.eval = Literal(node.val, node.dtype)
        else:
            node.val.start._gen_ir()
            node.val.stop._gen_ir()
            node.val.step._gen_ir()
            node.eval = Slice(node.val.start.eval, node.val.stop.eval, node.val.step.eval)


    elif type(node) == Var or (type(node) == Tensor and len(node._size()) == 0):
        node.eval = Scalar(node.dtype, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    elif type(node) == Tensor and len(node._size()) > 0:
        # convert AST sizes to IR sizes
        size = helpers.get_ir_of_size(node._size())
        node.eval = Ndarray(node.dtype, size, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    elif type(node) == TensorOp:
        if node.op_type in op_mapping or node.op_type in cmp_op:
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            assert isinstance(node.operators[0], Tensor) and isinstance(node.operators[1], Tensor)
            if is_same_size(node.operators[0]._size(), node.operators[1]._size()):
                if len(node._size()) > 0:
                    size = helpers.get_ir_of_size(node._size())
                    node.eval = Ndarray(node.dtype, size)
                    node.decl = [Decl(node.eval)]
                    pre_loop = Loop(0, node.eval.size[0], 1, [])
                    node.compute = [pre_loop]
                    lhs = bind(node.operators[0].eval, pre_loop.iterate)
                    rhs = bind(node.operators[1].eval, pre_loop.iterate)
                    res = bind(node.eval, pre_loop.iterate)
                    for i in range(1, len(node.eval.size)):
                        loop = Loop(0, node.eval.size[i], 1, [])
                        pre_loop.body.append(loop)
                        pre_loop = loop
                        lhs = bind(lhs, pre_loop.iterate)
                        rhs = bind(rhs, pre_loop.iterate)
                        res = bind(res, pre_loop.iterate)

                    if node.op_type in op_mapping:
                        op = op_mapping[node.op_type]
                    else:
                        op = node.op_type
                    assign = Assignment(res, Expr(lhs, rhs, op))
                    pre_loop.body.append(assign)

                else:
                    node.eval = Scalar(node.dtype)
                    node.decl = [Decl(node.eval)]
                    if node.op_type in op_mapping:
                        op = op_mapping[node.op_type]
                    else:
                        op = node.op_type
                    node.compute = [Assignment(node.eval, Expr(node.operators[0].eval, node.operators[1].eval, op))]
            else:
                size = helpers.get_ir_of_size(node._size())
                node.eval = Ndarray(node.dtype, size)
                node.decl = [Decl(node.eval)]
                pre_loop = Loop(0, node.eval.size[0], 1, [])
                node.compute = [pre_loop]
                lhs = bind(node.operators[0].eval, pre_loop.iterate)
                rhs = node.operators[1].eval
                res = bind(node.eval, pre_loop.iterate)
                for i in range(1, len(node.eval.size)):
                    loop = Loop(0, node.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    lhs = bind(lhs, pre_loop.iterate)
                    res = bind(res, pre_loop.iterate)

                if node.op_type in op_mapping:
                    op = op_mapping[node.op_type]
                else:
                    op = node.op_type
                assign = Assignment(res, Expr(lhs, rhs, op))
                pre_loop.body.append(assign)

        elif node.op_type in math_op:
            node.operators[0]._gen_ir()
            if len(node._size()) > 0:
                size = helpers.get_ir_of_size(node._size())
                node.eval = Ndarray(node.dtype, size)
                node.decl = [Decl(node.eval)]
                pre_loop = Loop(0, node.eval.size[0], 1, [])
                node.compute = [pre_loop]
                val = bind(node.operators[0].eval, pre_loop.iterate)
                res = bind(node.eval, pre_loop.iterate)
                for i in range(1, len(node.eval.size)):
                    loop = Loop(0, node.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    val = bind(val, pre_loop.iterate)
                    res = bind(res, pre_loop.iterate)

                assign = Assignment(res, Math(val, node.op_type))
                pre_loop.body.append(assign)

            else:
                node.eval = Scalar(node.dtype)
                node.decl = [Decl(node.eval)]
                node.compute = [Assignment(node.eval, Math(node.operators[0].eval, node.op_type))]

        elif node.op_type == 'setval':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()

            node.eval = node.operators[0].eval
            node.decl = node.operators[0].decl[:]
            node.operators[0].decl.clear()
            val = node.operators[1].eval

            if len(node.ref_size) > 0:
                size = helpers.get_ir_of_size(node.ref_size)
                pre_loop = Loop(0, size[0], 1, [])
                node.compute = [pre_loop]
                res = bind(node.eval, pre_loop.iterate)
                for i in range(1, len(size)):
                    loop = Loop(0, size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    res = bind(res, pre_loop.iterate)

                assign = Assignment(res, val)
                pre_loop.body.append(assign)

            else:
                node.compute = [Assignment(node.eval, val)]


        elif node.op_type == 'einsum':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            exp = node.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            all_indices = ''.join(sorted(set(input1 + input2)))
            all_loops = []
            for i in all_indices:
                pos1 = input1.find(i)
                if pos1 >= 0:
                    all_loops.append(Loop(0, node.operators[0].eval.size[pos1], 1, []))
                else:
                    pos2 = input2.find(i)
                    if pos2 >= 0:
                        all_loops.append(Loop(0, node.operators[1].eval.size[pos2], 1, []))
                    else:
                        raise IndexError('index not found!')

            op1 = node.operators[0].eval
            for i in input1:
                idx = all_indices.find(i)
                op1 = bind(op1, all_loops[idx].iterate)

            op2 = node.operators[1].eval
            for i in input2:
                idx = all_indices.find(i)
                op2 = bind(op2, all_loops[idx].iterate)

            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size, val=0)
            node.decl = [Decl(node.eval)]
            res = node.eval
            for i in output:
                idx = all_indices.find(i)
                res = bind(res, all_loops[idx].iterate)

            body = Assignment(res, Expr(op1, op2, '*'), '+')
            pre_loop = all_loops[0]
            node.compute = [pre_loop]
            for i in range(1, len(all_loops)):
                loop = all_loops[i]
                pre_loop.body.append(loop)
                pre_loop = loop
            pre_loop.body.append(body)


        elif node.op_type == 'index':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            if type(node.operators[1].eval) in (Scalar, Literal, Indexing):
                node.eval = Indexing(node.operators[0].eval, node.operators[1].eval)
            elif type(node.operators[1].eval) in (Ndarray, Slice):
                node.eval = Indexing(node.operators[0].eval, Indexing(node.operators[1].eval, Literal(-1, 'int')))
            else:
                raise TypeError('incorrect index type!')

        elif node.op_type == 'apply':

            node.operators[0]._gen_ir() # input tensor
            node.operators[2]._gen_ir() # axis

            axis = node.operators[2].eval.val

            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            # item is an indexing to the input tensor in axis dimension
            item = node.operators[3]
            item.eval = node.operators[0].eval
            for i in range(axis):
                item.eval = Indexing(item.eval, Literal(-1, 'int'))
            item.eval = Indexing(item.eval, outer_loop.iterate)

            ret = node.operators[-1]
            ret._gen_ir() # generate IR for applied func

            def action(node, res):
                if node.valid == True:
                    if type(node) == Var or type(node) == Tensor:
                        res.extend(node.decl)
                        node.valid = False
                    elif type(node) == TensorOp:
                        res.extend(node.decl)
                        res.extend(node.compute)
                        node.valid = False

            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            outer_loop.body.extend(ret_compute)
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))
            node.decl.extend(ret_decl)
            node.compute = [outer_loop]

            res = bind(node.eval, outer_loop.iterate)
            replace_output(node.compute, ret.eval, res)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

        elif node.op_type == 'reduce':
            node.operators[0]._gen_ir()
            node.operators[3]._gen_ir()
            axis = node.operators[3].eval.val

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)
            node.decl.append(Decl(node.eval))

            node.operators[2]._gen_ir() # init

            node.compute = []

            # TODO: iterating over the reduction dimension in the outer loop may not give best performance
            # TODO: it might be better to make it the innermost loop
            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            item1 = node.operators[4]
            item2 = node.operators[5]
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = Indexing(item2.eval, Literal(-1, 'int'))
            item2.eval = Indexing(item2.eval, outer_loop.iterate)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            ret._gen_ir()

            def action(node, res):
                if node.valid == True:
                    if type(node) == Var or type(node) == Tensor:
                        res.extend(node.decl)
                        node.valid = False
                    elif type(node) == TensorOp:
                        res.extend(node.decl)
                        res.extend(node.compute)
                        node.valid = False

            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            outer_loop.body.extend(ret_compute)
            node.decl.extend(ret_decl)
            node.compute.append(outer_loop)

            replace_output(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


        elif node.op_type == 'aggr':
            node.operators[0]._gen_ir() # input tensor
            node.operators[3]._gen_ir() # indices
            node.operators[4]._gen_ir() # axis
            axis = node.operators[4].eval.val
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.decl.append(Decl(node.eval))
            # this must be called after node.eval is constructed
            node.operators[2]._gen_ir() # init
            node.compute = []
            # compute
            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            item1 = node.operators[6]
            item2 = node.operators[7]
            item1.eval = Indexing(node.eval, Indexing(node.operators[3].eval, outer_loop.iterate))
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = Indexing(item2.eval, Literal(-1, 'int'))
            item2.eval = Indexing(item2.eval, outer_loop.iterate)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            ret._gen_ir()

            def action(node, res):
                if node.valid == True:
                    if type(node) == Var or type(node) == Tensor:
                        res.extend(node.decl)
                        node.valid = False
                    elif type(node) == TensorOp:
                        res.extend(node.decl)
                        res.extend(node.compute)
                        node.valid = False

            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            outer_loop.body.extend(ret_compute)
            node.decl.extend(ret_decl)
            node.compute.append(outer_loop)

            replace_output(node.compute, ret.eval, item1.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

    # points from IR back to ASTNode
    for d in node.decl:
        d.astnode = node
    for s in node.compute:
        s.astnode = node

    return node

