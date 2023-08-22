import torch

from core.ast import *
from core.ir import *
from opt.loop import *


def bind(arr: (Ndarray, Index), index, nref=False):
    if type(arr) == Ndarray or index == None or nref:
        return Index(arr, index=index)
    else:
        ref_chain = [arr]
        while (type(ref_chain[-1].dobject) != Ndarray):
            ref_chain.append(ref_chain[-1].dobject)
        for ref in ref_chain[::-1]:
            if ref.index == None:
                ref.index = index
                return arr
        return Index(arr, index=index)


def get_ir_of_size(size):
    ir_size = []
    for s in size:
        assert isinstance(s, ASTNode)
        gen_ir(s)
        ir_size.append(s.eval)
    return ir_size



def gen_ir(node):
    assert isinstance(node, ASTNode)
    if len(node.compute) > 0 or len(node.decl) > 0 or node.eval:
        return node
    if type(node) == Const:
        if node.dtype != 'slice':
            node.eval = node.val
        else:
            gen_ir(node.val.start)
            gen_ir(node.val.stop)
            gen_ir(node.val.step)
            node.eval = Slice(node.val.start.eval, node.val.stop.eval, node.val.step.eval)

    elif type(node) == Var or (type(node) == Tensor and len(node._size()) == 0):
        node.eval = Scalar(node.dtype, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    elif type(node) == Tensor and len(node._size()) > 0:
        # convert AST sizes to IR sizes
        size = get_ir_of_size(node._size())
        node.eval = Ndarray(node.dtype, size, node.name, node.is_arg)
        node.decl = [Decl(node.eval)]

    # here we define two special tensors to simply programming for sum/prod operations
    elif (type(node) == Ones or type(node) == Zeros) and len(node._size()) > 0:
        size = get_ir_of_size(node._size())
        node.eval = Ndarray(node.dtype, size, node.name, False, 1 if (type(node) == Ones) else 0)
        node.decl = [Decl(node.eval)]

    elif ((type(node) == Ones or type(node) == Zeros) and len(node._size()) == 0) or ((type(node) == One or type(node) == Zero)):
        node.eval = Scalar(node.dtype, node.name, False, 1 if (type(node) == Ones or type(node) == One) else 0)
        node.decl = [Decl(node.eval)]

    # elif type(node) == Set:
    #     gen_ir(node.storage)
    #     gen_ir(node.nelem)
    #     ref = Ref(node.storage.eval)
    #     node.eval = ref
    #     node.decl = [Decl(ref)]

    elif type(node) == TensorOp:
        if node.op_type in op_mapping:
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])
            # TODO: add support for scalar + tensor
            assert isinstance(node.operators[0], Tensor) and isinstance(node.operators[1], Tensor)
            if is_same_size(node.operators[0]._size(), node.operators[1]._size()):
                if len(node._size()) > 0:
                    size = get_ir_of_size(node._size())
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

                    op = op_mapping[node.op_type]
                    assign = Assignment(res, Expr(lhs, rhs, op))
                    pre_loop.body.append(assign)

                else:
                    node.eval = Scalar(node.dtype)
                    node.decl = [Decl(node.eval)]
                    node.compute = [Assignment(node.eval, Expr(node.operators[0].eval, node.operators[1].eval, op_mapping[node.op_type]))]


        elif node.op_type == 'index':
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])
            if type(node.operators[1]) == Var or (type(node.operators[1]) == Const and node.operators[1].dtype == 'int'):
                node.eval = Index(node.operators[0].eval, index=node.operators[1].eval)
            else:
                node.eval = Index(node.operators[0].eval, ind_arr=node.operators[1].eval)

        elif node.op_type == 'apply':
            gen_ir(node.operators[0])
            gen_ir(node.operators[2])

            axis = node.operators[2].eval

            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])
            item = node.operators[3]
            item.eval = node.operators[0].eval
            for i in range(axis):
                item.eval = bind(item.eval, None)
            item.eval = bind(item.eval, outer_loop.iterate, True)

            item.decl = []
            ret = node.operators[1](item)
            gen_ir(ret)
            node.operators.append(ret)
            node.dtype = ret.dtype
            node.ref_size = [node._size()[0]] + ret._size()
            node.fix_size = []
            outer_loop.body.extend(ret.compute[:])
            ret.compute.clear()
            size = get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))

            # node.eval <= ret.eval
            res = bind(node.eval, outer_loop.iterate)
            if (len(ret.eval.size) > 0):
                pre_loop = Loop(0, ret.eval.size[0], 1, [])
                outer_loop.body.append(pre_loop)
                res = bind(res, pre_loop.iterate)
                rhs = bind(ret.eval, pre_loop.iterate)
                for i in range(1, len(ret.eval.size)):
                    loop = Loop(0, ret.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    res = bind(res, pre_loop.iterate)
                    rhs = bind(rhs, pre_loop.iterate)
                pre_loop.body.append(Assignment(res, rhs))
            else:
                assign = Assignment(res, ret.eval)
                outer_loop.body.append(assign)

            scope = outer_loop.body
            while len(scope) == 2:
                fuse(scope, scope[0], scope[1])
                if type(scope[0]) is not Loop:
                    break
                scope = scope[0].body

            node.compute = [outer_loop]

        elif node.op_type == 'reduce':
            gen_ir(node.operators[0])
            gen_ir(node.operators[2])
            gen_ir(node.operators[3])
            axis = node.operators[3].eval

            size = get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = Ndarray(node.dtype, size)
            else:
                node.eval = Scalar(node.dtype)
            node.decl.append(Decl(node.eval))

            node.compute = []
            if len(node.eval.size) > 0:
                pre_loop = Loop(0, node.eval.size[0], 1, [])
                node.compute.append(pre_loop)
                res = bind(node.eval, pre_loop.iterate)
                rhs = bind(node.operators[2].eval, pre_loop.iterate)
                for i in range(1, len(node.eval.size)):
                    loop = Loop(0, node.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    res = bind(res, pre_loop.iterate)
                    rhs = bind(rhs, pre_loop.iterate)
                pre_loop.body.append(Assignment(res, rhs))
            else:
                assign = Assignment(node.eval, node.operators[2].eval)
                node.compute.append(assign)


            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            item1 = node.operators[4]
            item2 = node.operators[5]
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = bind(item2.eval, None)
            item2.eval = bind(item2.eval, outer_loop.iterate, True)
            item2.decl = []
            item1.decl = []

            ret = node.operators[1](item1, item2)
            gen_ir(ret)
            node.operators.append(ret)
            outer_loop.body.extend(ret.compute[:])
            ret.compute.clear()

            if (len(ret.eval.size) > 0):
                pre_loop = Loop(0, ret.eval.size[0], 1, [])
                outer_loop.body.append(pre_loop)
                res = bind(node.eval, pre_loop.iterate)
                rhs = bind(ret.eval, pre_loop.iterate)
                for i in range(1, len(ret.eval.size)):
                    loop = Loop(0, ret.eval.size[i], 1, [])
                    pre_loop.body.append(loop)
                    pre_loop = loop
                    res = bind(res, pre_loop.iterate)
                    rhs = bind(rhs, pre_loop.iterate)
                pre_loop.body.append(Assignment(res, rhs))
            else:
                assign = Assignment(node.eval, ret.eval)
                outer_loop.body.append(assign)

            node.compute.append(outer_loop)

    return node

