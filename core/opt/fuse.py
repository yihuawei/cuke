from core.asg import *
from core.ir import *
import codegen
import helpers
from core.opt.reorder import rebind_iterate

def get_obj(ir: (Indexing, Scalar)):
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
        if type(ir.left) in (Indexing, Scalar):
            obj = get_obj(ir.left)
            if obj == old:
                ir.left = new
        else:
            replace_index_with_scalar(ir.left, old, new)
        if type(ir.right) in (Indexing, Scalar):
            obj = get_obj(ir.right)
            if obj == old:
                ir.right = new
        else:
            replace_index_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Indexing, Scalar):
            obj = get_obj(ir.lhs)
            if obj == old:
                ir.lhs = new
        else:
            replace_index_with_scalar(ir.lhs, old, new)
        if type(ir.rhs) in (Indexing, Scalar):
            obj = get_obj(ir.rhs)
            if obj == old:
                ir.rhs = new
        else:
            replace_index_with_scalar(ir.rhs, old, new)
    elif type(ir) == Slice:
        if type(ir.start) in (Indexing, Scalar):
            obj = get_obj(ir.start)
            if obj == old:
                ir.start = new
        else:
            replace_index_with_scalar(ir.start, old, new)

        if type(ir.stop) in (Indexing, Scalar):
            obj = get_obj(ir.stop)
            if obj == old:
                ir.stop = new
        else:
            replace_index_with_scalar(ir.stop, old, new)

        if type(ir.step) in (Indexing, Scalar):
            obj = get_obj(ir.step)
            if obj == old:
                ir.step = new
        else:
            replace_index_with_scalar(ir.step, old, new)

    elif type(ir) == Math:
        if type(ir.val) in (Indexing, Scalar):
            obj = get_obj(ir.val)
            if obj == old:
                ir.val = new
        else:
            replace_index_with_scalar(ir.val, old, new)


def fuse(node, fusion_type = ['basic']):
    binary_elw =  list(arith_op.keys()) + cmp_op
    unary_elw = math_op
    elementwise_op = binary_elw + unary_elw

    for ft in fusion_type:
        if ft == 'basic':
            def action(node, res):
                if node.valid == True:
                    if type(node) == TensorOp and node.op_type in elementwise_op:
                        if type(node.operators[0]) == TensorOp and len(node.operators[0].ref_by) == 1:
                            # the input is always consumed in the last statement in the loop body for arith_op, math_op, and cmp_op
                            assign = node.input_orders[0][-1][1].body[-1]
                            new_term = None
                            if (node.operators[0].op_type in (elementwise_op + ['setval', 'apply'])):
                                new_term = node.operators[0].output_order[-1][1].body[-1].rhs
                                node.input_orders[0][-1][1].body[0:0] = node.operators[0].output_order[-1][1].body[:len(node.operators[0].output_order[-1][1].body)-1]
                            elif node.operators[0].op_type in ['einsum', 'reduce']:
                                new_term = Scalar(node.operators[0].eval.dtype)
                                node.decl.append(Decl(new_term))
                                replace_index_with_scalar(node.operators[0].output_order[-1][1].body, node.operators[0].eval, new_term)
                                node.input_orders[0][-1][1].body[0:0] = node.operators[0].output_order[-1][1].body[:]
                            if new_term != None:
                                if node.op_type in math_op:
                                    assign.rhs.val = new_term
                                else:
                                    assign.rhs.left = new_term

                                for i in range(len(node.input_orders[0])):
                                    nl = node.input_orders[0][i][1]
                                    ol = node.operators[0].output_order[i][1]
                                    rebind_iterate(node.input_orders[0][-1][1].body, ol.iterate, nl.iterate)
                                    if 'loop_ofs' in ol.attr:
                                        if 'loop_ofs' in nl.attr:
                                            nl.attr['loop_ofs'] = max(nl.attr['loop_ofs'], ol.attr['loop_ofs'])
                                        else:
                                            nl.attr['loop_ofs'] = ol.attr['loop_ofs']

                                node.operators[0].decl = [d for d in node.operators[0].decl if get_obj(d) != node.operators[0].eval]
                                node.decl.extend(node.operators[0].decl)
                                node.operators[0].valid = False

                        if node.op_type in binary_elw:
                            if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
                                assign = node.input_orders[1][-1][1].body[-1]
                                new_term = None
                                if (node.operators[1].op_type in (elementwise_op + ['setval', 'apply'])):
                                    new_term = node.operators[1].output_order[-1][1].body[-1].rhs
                                    node.input_orders[1][-1][1].body[-1:-1] = node.operators[1].output_order[-1][1].body[:len(node.operators[1].output_order[-1][1].body)-1]
                                elif node.operators[1].op_type in ['einsum', 'reduce']:
                                    new_term = Scalar(node.operators[1].eval.dtype)
                                    node.decl.append(Decl(new_term))
                                    replace_index_with_scalar(node.operators[1].output_order[-1][1].body, node.operators[1].eval, new_term)
                                    node.input_orders[1][-1][1].body[-1:-1] = node.operators[1].output_order[-1][1].body[:]
                                if new_term != None:
                                    if node.op_type in math_op:
                                        assign.rhs.val = new_term
                                    else:
                                        assign.rhs.right = new_term

                                    for i in range(len(node.input_orders[1])):
                                        nl = node.input_orders[1][i][1]
                                        ol = node.operators[1].output_order[i][1]
                                        rebind_iterate(node.input_orders[1][-1][1].body, ol.iterate, nl.iterate)
                                        if 'loop_ofs' in ol.attr:
                                            if 'loop_ofs' in nl.attr:
                                                nl.attr['loop_ofs'] = max(nl.attr['loop_ofs'], ol.attr['loop_ofs'])
                                            else:
                                                nl.attr['loop_ofs'] = ol.attr['loop_ofs']

                                    node.operators[1].decl = [d for d in node.operators[1].decl if get_obj(d) != node.operators[1].eval]
                                    node.decl.extend(node.operators[1].decl)
                                    node.operators[1].valid = False

            t = helpers.Traversal(action)
            t(node)

    return node


def test1():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (10, 20))
    C = Tensor('c', (10, 20))
    D = Tensor('d', (10, 20))

    A = setval(A, 1)
    t1 = A + B
    t2 = (C - D).abs()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    code = codegen.cpu.print_cpp(fuse(ir1))
    print(code)


def test2():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 30))
    D = Tensor('d', (10, 30))
    t1 = (A @ B).abs()
    t2 = (C - D).abs()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    code = codegen.cpu.print_cpp(fuse(ir1))
    print(code)

def test3():
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 20))
    D = Tensor('d', (20, 30))
    t1 = (A @ B).abs()
    t2 = (C @ D).round()
    res1 = t1 + t2
    ir1 = res1._gen_ir()
    fuse(ir1)
    code = codegen.cpu.print_cpp(ir1)
    print(code)

def compression():
    input = Tensor('input', (50, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res.apply(lambda x:x[0:32]-x[-1:31], axis=0)
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum()
    compressed_res = apply(lambda x, y: x // y,lorenzo_res, encode_nbits, out_ofs=ofs)
    res = compressed_res
    code = codegen.cpu.print_cpp(fuse(res._gen_ir()))
    print(code)

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    compression()