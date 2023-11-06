from core.ast import *
from core.ir import *
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
    if node.eval or len(node.decl) > 0 or (type(node) == TensorOp and len(node.compute) > 0):
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
        if node.op_type in arith_op or node.op_type in cmp_op:
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            node.input_orders[0] = []
            node.input_orders[1] = []
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

                    if node.op_type in arith_op:
                        op = arith_op[node.op_type]
                    else:
                        op = node.op_type
                    assign = Assignment(res, Expr(lhs, rhs, op))
                    pre_loop.body.append(assign)

                else:
                    node.eval = Scalar(node.dtype)
                    node.decl = [Decl(node.eval)]
                    if node.op_type in arith_op:
                        op = arith_op[node.op_type]
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
                rhs_level = 1
                res = bind(node.eval, pre_loop.iterate)
                for i in range(1, len(node.eval.size)):
                    loop = Loop(0, node.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    lhs = bind(lhs, pre_loop.iterate)
                    if rhs_level < len(node.operators[1].ref_size):
                        rhs = bind(rhs, pre_loop.iterate)
                        rhs_level += 1
                    res = bind(res, pre_loop.iterate)

                if node.op_type in arith_op:
                    op = arith_op[node.op_type]
                else:
                    op = node.op_type
                assign = Assignment(res, Expr(lhs, rhs, op))
                pre_loop.body.append(assign)

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                node.input_orders[0].append((i, l))
                node.input_orders[1].append((i, l))
                l = l.body[0]



        elif node.op_type in math_op:
            node.operators[0]._gen_ir()
            node.input_orders[0] = []
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

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                node.input_orders[0].append((i, l))
                l = l.body[0]

        elif node.op_type == 'setval':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            node.input_orders[0] = []
            # node.operators[1] must be a Scalar, so no input_order is needed

            node.eval = node.operators[0].eval
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

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                node.input_orders[0].append((i, l))
                l = l.body[0]


        elif node.op_type == 'einsum':
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            node.input_orders[0] = []
            node.input_orders[1] = []

            exp = node.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            all_indices = ''.join(sorted(set(input1 + input2)))
            all_loops = []
            mapping = {}
            for i in output:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[0].eval.size[pos1], 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = Loop(0, node.operators[1].eval.size[pos2], 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            reduce_begins = len(all_loops)

            for i in range(len(all_indices)):
                pos1 = input1.find(all_indices[i])
                pos2 = input2.find(all_indices[i])
                if pos1 >= 0 and pos2 >= 0:
                    mapping[all_indices[i]] = len(all_loops)
                    l = Loop(0, node.operators[0].eval.size[pos1], 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 < 0 and pos2 < 0:
                    raise IndexError('index not found!')

            op1 = node.operators[0].eval
            for i in input1:
                op1 = bind(op1, all_loops[mapping[i]].iterate)

            op2 = node.operators[1].eval
            for i in input2:
                op2 = bind(op2, all_loops[mapping[i]].iterate)

            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.decl = [Decl(node.eval)]
            res = node.eval
            for i in output:
                res = bind(res, all_loops[mapping[i]].iterate)

            if reduce_begins == len(all_loops):
                body = Assignment(res, Expr(op1, op2, '*'))
            else:
                body = Assignment(res, Expr(op1, op2, '*'), '+')
            init = Assignment(res, 0)
            if reduce_begins == 0:
                node.compute.append(init)
            pre_loop = all_loops[0]
            node.compute.append(pre_loop)
            for i in range(1, len(all_loops)):
                if reduce_begins == i:
                    pre_loop.body.append(init)
                loop = all_loops[i]
                pre_loop.body.append(loop)
                pre_loop = loop
            pre_loop.body.append(body)

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                l = l.body[0]


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
            #TODO: add input_orders for apply, reduce, and aggr
            func = node.operators[0]
            nparams = len(inspect.signature(func).parameters)

            for i in range(2, 2+2*nparams):
                node.operators[i]._gen_ir()

            primary_axis = node.operators[2+nparams].eval.val

            outer_loop = Loop(0, node.operators[2].eval.size[primary_axis], 1, [])

            for i in range(nparams):
                item = node.operators[2+2*nparams+i]
                item.eval = node.operators[2+i].eval
                axis = node.operators[2+nparams+i].eval.val
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

            if len(ret_compute) == 0:
                ret_compute.append(Assignment(ret.eval, ret.eval))

            out_ofs = node.operators[1]
            if out_ofs != None:
                out_ofs._gen_ir()
                real_loop = Loop(Indexing(out_ofs.eval, outer_loop.iterate), Indexing(out_ofs.eval, Expr(outer_loop.iterate, 1, '+')), 1, [])
                outer_loop.body = [real_loop]
            else:
                real_loop = outer_loop
            real_loop.body.extend(ret_compute)
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))
            node.decl.extend(ret_decl)
            node.compute = [outer_loop]

            res = bind(node.eval, real_loop.iterate)
            replace_output(node.compute, ret.eval, res)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            # TODO: need test for this
            node.output_order = [(0, outer_loop)]
            if hasattr(ret, 'output_order'):
                for i in range(len(ret.output_order)):
                    node.output_order.append((i+1, ret.output_order[i][1]))


        elif node.op_type == 'reduce':
            node.operators[0]._gen_ir()
            node.operators[3]._gen_ir()
            axis = node.operators[3].eval.val

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)

            node.operators[2]._gen_ir() # init
            # the decl of node.eval should be added to the init
            node.operators[2].decl.append(Decl(node.eval))


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

            node.output_order = ret.output_order

        elif node.op_type == 'scan':
            node.operators[0]._gen_ir()
            node.operators[2]._gen_ir()
            axis = node.operators[2].eval.val

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)

            ninits = (len(node.operators) - 5) // 2

            for init in node.operators[3:3+ninits]:
                init._gen_ir() # init
            node.operators[3].decl.append(Decl(node.eval))

            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            item = node.operators[4]
            item.eval = node.operators[0].eval
            for i in range(axis):
                item.eval = Indexing(item.eval, Literal(-1, 'int'))
            item.eval = Indexing(item.eval, outer_loop.iterate)
            item.decl = []

            ys = node.operators[5:5+ninits]
            for i in range(len(ys)):
                ys[i].eval = node.eval
                ys[i].eval = Indexing(ys[i].eval, Expr(outer_loop.iterate, Literal(i, 'int'), '+'))
                ys[i].decl = []

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

            replace_output(node.compute, ret.eval, Indexing(node.eval, Expr(outer_loop.iterate, Literal(len(ys), 'int'), '+')))
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            # TODO: need testing
            node.output_order = [(0, outer_loop)]
            for i in range(len(ret.output_order)):
                node.output_order.append((i+1, ret.output_order[i][1]))

        elif node.op_type == 'aggr':
            node.operators[0]._gen_ir() # input tensor
            node.operators[3]._gen_ir() # indices
            node.operators[4]._gen_ir() # axis
            axis = node.operators[4].eval.val
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.operators[2]._gen_ir() # init
            node.operators[2].decl.append(Decl(node.eval))

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


            node.output_order = [(0, outer_loop)]
            for i in range(len(ret.output_order)):
                node.output_order.append((i+1, ret.output_order[i][1]))

    # points from IR back to ASTNode
    for d in node.decl:
        d.astnode = node

    if type(node) == TensorOp:
        for s in node.compute:
            s.astnode = node

    return node

