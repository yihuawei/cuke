import copy

import codegen.cpu
from core.asg import *
from core.ir import *
import helpers
from core.opt.reorder import rebind_iterate



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
        old = copy.copy(x.idx)
        x.idx = idx
        new_index = copy.deepcopy(index)
        x.idx = old
        return new_index


def get_slice(index: (Indexing, Ndarray, Slice)):
    if type(index) == Indexing:
        x = get_slice(index.dobject)
        if x != None:
            return x
        else:
            y = get_slice(index.idx)
            if y != None:
                return y
            else:
                if type(index.dobject) == Slice and type(index.idx) == Literal and index.idx.val == -1:
                    return index.dobject
    return None


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

def has_same_iteration_space(l1, l2):
    return has_same_value(l1.start, l2.start) and has_same_value(l1.end, l2.end) and has_same_value(l1.step, l2.step)


def gen_binary_op(left, right, res, op, ir, level, left_max_level, right_max_level, res_size):

    max_level = max(left_max_level, right_max_level)
    if level == max_level:
        assign = Assignment(res, Expr(left, right, op))
        ir.append(assign)
    else:
        ssl = get_slice(left)
        ssr = get_slice(right)
        if ssl != None and type(ssl.start) == Literal:
            ssl = ssl.start.val
            if ssl < 0:
                ssl = 0 - ssl
            else:
                ssl = 0
        else:
            ssl = 0

        if ssr != None and type(ssr.start) == Literal:
            ssr = ssr.start.val
            if ssr < 0:
                ssr = 0 - ssr
            else:
                ssr = 0
        else:
            ssr = 0

        ssi = min(ssl, ssr)

        for i in range(ssi):
            lhs1 = Literal('0', dtype=left.dtype)
            rhs1 = Literal('0', dtype=right.dtype)
            res1 = bind(res, Literal(i, 'int'))
            gen_binary_op(lhs1, rhs1, res1, op, ir, level+1, left_max_level, right_max_level, res_size)

        for i in range(ssi, ssl):
            lhs1 = Literal('0', dtype=left.dtype)
            if type(right) != Literal:
                rhs1 = bind(right, Literal(i - ssi, 'int'))
            else:
                rhs1 = right
            res1 = bind(res, Literal(i, 'int'))
            gen_binary_op(lhs1, rhs1, res1, op, ir, level+1, left_max_level, right_max_level, res_size)

        for i in range(ssi, ssr):
            if type(left) != Literal:
                lhs1 = bind(left, Literal(i - ssi, 'int'))
            else:
                lhs1 = left
            rhs1 = Literal('0', dtype=right.dtype)
            res1 = bind(res, Literal(i, 'int'))
            gen_binary_op(lhs1, rhs1, res1, op, ir, level+1, left_max_level, right_max_level, res_size)

        ssa = max(ssl, ssr)

        pre_loop = Loop(ssa, res_size[level], 1, [])
        if level < left_max_level and type(left) != Literal:
            lhs = bind(left, pre_loop.iterate)
        else:
            lhs = left
        if level < right_max_level and type(right) != Literal:
            rhs = bind(right, pre_loop.iterate)
        else:
            rhs = right
        res = bind(res, pre_loop.iterate)

        gen_binary_op(lhs, rhs, res, op, pre_loop.body, level+1, left_max_level, right_max_level, res_size)
        ir.append(pre_loop)



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

            if node.op_type in arith_op:
                op = arith_op[node.op_type]
            else:
                op = node.op_type
            if len(node._size()) > 0:
                size = helpers.get_ir_of_size(node._size())
                node.eval = Ndarray(node.dtype, size)
            else:
                size = []
                node.eval = Scalar(node.dtype)
            node.decl = [Decl(node.eval)]

            gen_binary_op(node.operators[0].eval, node.operators[1].eval, node.eval, op, node.compute, 0, len(node.operators[0]._size()), len(node.operators[1]._size()), size)

            # TODO: handle negative slice
            # l = node.compute[0]
            # for i in range(len(node.eval.size)):
            #     node.output_order.append((i, l))
            #     node.input_orders[0].append((i, l))
            #     node.input_orders[1].append((i, l))
            #     l = l.body[0]



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
            if type(node.operators[0]) == Tensor:
                node.operators[0].is_arg = False

            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()

            node.eval = node.operators[0].eval

            if is_scalar(node.operators[1]):
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
                    l = l.body[0]

            else:
                node.decl = [d for d in node.operators[1].decl if d.dobject != node.operators[1].eval]
                node.compute = node.operators[1].compute
                replace_output(node.compute, node.operators[1].eval, node.eval)
                node.operators[1].valid = False


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

            # generate IR for the input items of func
            for i in range(2, 2+2*nparams):
                node.operators[i]._gen_ir()

            primary_axis = node.operators[2+nparams].eval.val

            # this is the loop that iterates over the axis of the primary (first) tensor input
            outer_loop = Loop(0, node.operators[2].eval.size[primary_axis], 1, [])

            for i in range(nparams):
                item = node.operators[2+2*nparams+i]
                item.eval = node.operators[2+i].eval
                axis = node.operators[2+nparams+i].eval.val
                for i in range(axis):
                    item.eval = Indexing(item.eval, Literal(-1, 'int'))
                item.eval = Indexing(item.eval, outer_loop.iterate)

            # since input items of func has been generated and indexed, we can generate the IR of the func
            ret = node.operators[-1]
            ret._gen_ir()

            def action(node, res):
                if node.valid == True:
                    if isinstance(node, Tensor):
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

            # if there is no compute in the func, we simply assign the result to itself, so that later the lhs of the assignment will be changed to the output array
            if len(ret_compute) == 0:
                ret_compute.append(Assignment(ret.eval, ret.eval))

            # the compute of func are added in the outer_loop
            outer_loop.body.extend(ret_compute)
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))
            node.decl.extend(ret_decl)
            node.compute = [outer_loop]

            out_ofs = node.operators[1]
            res = bind(node.eval, outer_loop.iterate) if out_ofs == None else node.eval
            replace_output(node.compute, ret.eval, res)
            # if there is an offset for output storage
            if out_ofs != None:
                # the last statement in the func IR is always a Loop that writes the result to ret.eval (which has been replaced by res)
                assert type(ret.compute[-1]) == Loop
                # But the index to the node.eval in res is incorrect, we need to change it according to the offset
                rebind_iterate(ret.compute[-1].body[-1].lhs, ret.compute[-1].iterate, Expr(Indexing(out_ofs.eval, outer_loop.iterate), ret.compute[-1].iterate, '+'))
            # ret.eval is removed from the decl
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

            compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute
            outer_loop.body = compute[:]
            compute.clear()

            # merge init into node.compute
            init = node.operators[2].output_order[-1][1].body if len(node.operators[2].output_order) > 0 else node.operators[2].compute
            # assert len(node.operators[2].output_order) == len(ret.output_order)
            for i in range(len(node.operators[2].output_order)):
                # assert has_same_iteration_space(node.operators[2].output_order[i][1], ret.output_order[i][1])
                rebind_iterate(init, node.operators[2].output_order[i][1].iterate, ret.output_order[i][1].iterate)
                node.output_order.append((i, ret.output_order[i][1]))
            compute.extend(init)
            node.operators[2].valid = False
            compute.append(outer_loop)

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

            node.decl.extend(ret_decl)
            node.compute.extend(ret_compute)

            replace_output(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


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

