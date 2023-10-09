from core.ast import *
from core.ir import *
from opt.loop import *
import helpers



def bind(arr: (Ndarray, Index), index, fix_ref=False):
    # fix_ref == True means index at current position instead of the first unbind
    if type(arr) == Ndarray or index == None or fix_ref:
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






def gen_ir(node):
    assert isinstance(node, ASTNode)
    if len(node.compute) > 0 or len(node.decl) > 0 or node.eval:
        return node
    if type(node) == Const:
        if node.dtype != 'slice':
            node.eval = node.val
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

    # here we define two special tensors to simply programming for sum/prod operations
    elif (type(node) == Ones or type(node) == Zeros) and len(node._size()) > 0:
        size = helpers.get_ir_of_size(node._size())
        node.eval = Ndarray(node.dtype, size, node.name, False, 1 if (type(node) == Ones) else 0)
        node.decl = [Decl(node.eval)]

    elif ((type(node) == Ones or type(node) == Zeros) and len(node._size()) == 0) or ((type(node) == One or type(node) == Zero)):
        node.eval = Scalar(node.dtype, node.name, False, 1 if (type(node) == Ones or type(node) == One) else 0)
        node.decl = [Decl(node.eval)]

    elif type(node) == TensorOp:
        if node.op_type in op_mapping:
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            # TODO: add support for scalar + tensor
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

                    op = op_mapping[node.op_type]
                    assign = Assignment(res, Expr(lhs, rhs, op))
                    pre_loop.body.append(assign)

                else:
                    node.eval = Scalar(node.dtype)
                    node.decl = [Decl(node.eval)]
                    node.compute = [Assignment(node.eval, Expr(node.operators[0].eval, node.operators[1].eval, op_mapping[node.op_type]))]

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
                op1 = bind(op1, all_loops[idx].iterate, )

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
            if type(node.operators[1]) == Var or (type(node.operators[1]) == Const and node.operators[1].dtype == 'int'):
                node.eval = Index(node.operators[0].eval, index=node.operators[1].eval)
            else: # ind_arr can be a slice or Tensor
                # TODO: asssert tensor is 1d int?
                node.eval = Index(node.operators[0].eval, ind_arr=node.operators[1].eval)

        elif node.op_type == 'apply':

            node.operators[0]._gen_ir()
            node.operators[2]._gen_ir()

            axis = node.operators[2].eval

            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])
            item = node.operators[3]
            item.eval = node.operators[0].eval
            for i in range(axis):
                item.eval = bind(item.eval, None)
            item.eval = bind(item.eval, outer_loop.iterate, True)

            item.decl = []
            ret = node.operators[1](item)
            ret._gen_ir()

            def action(node, res):
                if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                    res.extend(node.decl)
                    # we do not call node.decl.clear() because we want keep the pointers to the IR in the original ASTNode. This pointer is not scanned for code generation though as the ASTNode.eval will be set to None
                elif type(node) == TensorOp:
                    res.extend(node.decl)
                    res.extend(node.compute)
                    # the same as above, we do not call node.decl.clear() or node.compute.clear()

            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            node.operators.append(ret)
            node.dtype = ret.dtype
            node.ref_size = [node._size()[0]] + ret._size()
            node.fix_size = []
            outer_loop.body.extend(ret_compute)
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(ret.eval.dtype, size)
            node.decl.append(Decl(node.eval))
            node.decl.extend(ret_decl)

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
            ret.valid = False

        elif node.op_type == 'reduce':
            node.operators[0]._gen_ir()
            node.operators[2]._gen_ir() # init
            node.operators[3]._gen_ir()
            axis = node.operators[3].eval

            size = helpers.get_ir_of_size(node._size())
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
            for i in range(axis): # TODO: is this correct?
                item2.eval = bind(item2.eval, None)
            if type(item2.eval) == Index and type(item2.eval.ind_arr) == Slice:
                item2.eval = bind(item2.eval, outer_loop.iterate)
            else:
                item2.eval = bind(item2.eval, outer_loop.iterate, True)
            item2.decl = []
            item1.decl = []

            ret = node.operators[1](item1, item2)
            ret._gen_ir()

            def action(node, res):
                if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                    res.extend(node.decl)
                    # node.decl.clear()
                elif type(node) == TensorOp:
                    res.extend(node.decl)
                    res.extend(node.compute)
                    # node.decl.clear()
                    # node.compute.clear()


            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            node.operators.append(ret)
            outer_loop.body.extend(ret_compute)
            node.decl.extend(ret_decl)

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
            ret.valid = False

        elif node.op_type == 'aggr':
            node.operators[0]._gen_ir() # input tensor
            node.operators[2]._gen_ir() # init
            node.operators[3]._gen_ir() # indices
            node.operators[4]._gen_ir() # axis
            axis = node.operators[4].eval
            size = helpers.get_ir_of_size(node._size())
            node.eval = Ndarray(node.dtype, size)
            node.decl.append(Decl(node.eval))

            node.compute = []
            # initialize output
            pre_loop = Loop(0, node.eval.size[0], 1, [])
            node.compute.append(pre_loop)
            res = bind(node.eval, pre_loop.iterate)
            rhs = node.operators[2].eval
            for i in range(1, len(node.eval.size)):
                loop = Loop(0, node.eval.size[i], 1, [])
                pre_loop.body.append(loop)
                pre_loop = loop
                res = bind(res, pre_loop.iterate)
                rhs = bind(rhs, pre_loop.iterate)
            pre_loop.body.append(Assignment(res, rhs))

            # compute
            outer_loop = Loop(0, node.operators[0].eval.size[axis], 1, [])

            item1 = node.operators[6]
            item2 = node.operators[7]
            item1.eval = Index(node.eval, ind_arr=node.operators[3].eval)
            item1.eval = bind(item1.eval, outer_loop.iterate)
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = bind(item2.eval, None)
            item2.eval = bind(item2.eval, outer_loop.iterate, True)
            item2.decl = []
            item1.decl = []

            ret = node.operators[1](item1, item2)
            ret._gen_ir()

            def action(node, res):
                if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                    res.extend(node.decl)
                    # node.decl.clear()
                elif type(node) == TensorOp:
                    res.extend(node.decl)
                    res.extend(node.compute)
                    # node.decl.clear()
                    # node.compute.clear()


            t = helpers.Traversal(action)
            ret_ir = t(ret)
            ret_decl = []
            ret_compute = []

            for ir in ret_ir:
                if type(ir) == Decl:
                    ret_decl.append(ir)
                else:
                    ret_compute.append(ir)

            node.operators.append(ret)
            outer_loop.body.extend(ret_compute)
            node.decl.extend(ret_decl)

            if (len(ret.eval.size) > 0):
                pre_loop = Loop(0, ret.eval.size[0], 1, [])
                outer_loop.body.append(pre_loop)
                res = bind(Index(node.eval, ind_arr=node.operators[3].eval), outer_loop.iterate)
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
                res = bind(Index(node.eval, ind_arr=node.operators[3].eval), outer_loop.iterate)
                assign = Assignment(res, ret.eval)
                outer_loop.body.append(assign)

            node.compute.append(outer_loop)
            ret.valid = False

    # points from IR back to ASTNode
    for d in node.decl:
        d.astnode = node
    for s in node.compute:
        s.astnode = node

    return node

