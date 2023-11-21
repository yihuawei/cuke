from core.asg import *
from core.ir import *
import codegen
from helpers import get_obj, get_val, ASGTraversal, rebind_iterate, IRTraversal, ir_defs, ir_uses



class fuser:
    def __init__(self):
        self.rules = []

    def register(self, rule):
        self.rules.append(rule)

    def fuse(self, node):
        def action(node, res):
            for r in self.rules:
                r(node, res)

        t = ASGTraversal(action)
        t(node)
        return node


def _find_defs(ir, data):
    def action(stmt, res):
        if type(stmt) == Assignment and get_obj(stmt.lhs) == data:
            res.append(stmt)

        return [True, True, True, True]

    t = IRTraversal(action)
    return t(ir)


def _replace_arrindex_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            _replace_arrindex_with_scalar(l, old, new)
    elif type(ir) == Loop:
        _replace_arrindex_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Indexing, Scalar):
            obj = get_obj(ir.left)
            if obj == old:
                ir.left = new
        else:
            _replace_arrindex_with_scalar(ir.left, old, new)
        if type(ir.right) in (Indexing, Scalar):
            obj = get_obj(ir.right)
            if obj == old:
                ir.right = new
        else:
            _replace_arrindex_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Indexing, Scalar):
            obj = get_obj(ir.lhs)
            if obj == old:
                ir.lhs = new
        else:
            _replace_arrindex_with_scalar(ir.lhs, old, new)
        if type(ir.rhs) in (Indexing, Scalar):
            obj = get_obj(ir.rhs)
            if obj == old:
                ir.rhs = new
        else:
            _replace_arrindex_with_scalar(ir.rhs, old, new)
    elif type(ir) == Slice:
        if type(ir.start) in (Indexing, Scalar):
            obj = get_obj(ir.start)
            if obj == old:
                ir.start = new
        else:
            _replace_arrindex_with_scalar(ir.start, old, new)

        if type(ir.stop) in (Indexing, Scalar):
            obj = get_obj(ir.stop)
            if obj == old:
                ir.stop = new
        else:
            _replace_arrindex_with_scalar(ir.stop, old, new)

        if type(ir.step) in (Indexing, Scalar):
            obj = get_obj(ir.step)
            if obj == old:
                ir.step = new
        else:
            _replace_arrindex_with_scalar(ir.step, old, new)

    elif type(ir) == Math:
        if type(ir.val) in (Indexing, Scalar):
            obj = get_obj(ir.val)
            if obj == old:
                ir.val = new
        else:
            _replace_arrindex_with_scalar(ir.val, old, new)

def match_orders(order1, order2):
    if len(order1) == len(order2):
        for i in range(len(order1)):
            x1 = get_val(order1[i][1].start)
            y1 = get_val(order1[i][1].end)
            z1 = get_val(order1[i][1].step)
            x2 = get_val(order2[i][1].start)
            y2 = get_val(order2[i][1].end)
            z2 = get_val(order2[i][1].step)
            if x1 == None or (x1 != x2 and x1.dobject_id != x2.dobject_id):
                return False
            if y1 == None or (y1 != y2 and y1.dobject_id != y2.dobject_id):
                return False
            if z1 == None and (z1 != z2 and z1.dobject_id != z2.dobject_id):
                return False
        return True
    else:
        return False

def merge_loops(order1, order2, data, astnode):
    if match_orders(order1, order2):
        for i in range(len(order1)):
            nl = order1[i][1]
            ol = order2[i][1]
            rebind_iterate(order2[-1][1], ol.iterate, nl.iterate)
            if i < len(order1) - 1:
                nl.body[0:0] = [s for s in ol.body if s != order2[i+1][1]]
            if 'loop_ofs' in ol.attr:
                if 'loop_ofs' in nl.attr:
                    nl.attr['loop_ofs'] = max(nl.attr['loop_ofs'], ol.attr['loop_ofs'])
                else:
                    nl.attr['loop_ofs'] = ol.attr['loop_ofs']

        dfs = _find_defs(order2[-1][1].body, data)
        if len(dfs) < 1:
            return False

        if ir_uses(dfs[-1], data):
            df = Scalar(data.dtype)
            astnode.decl.append(Decl(df))
        else:
            df = dfs[-1].rhs
            order2[-1][1].body = [s for s in order2[-1][1].body if not ir_defs(s, data)]

        j = len(order1[-1][1].body)
        for i in range(len(order1[-1][1].body)):
            if ir_uses(order1[-1][1].body[i], data):
                j = i
                break
        order1[-1][1].body[j:j] = order2[-1][1].body

        _replace_arrindex_with_scalar(order1[-1][1].body, data, df)
        return True
    return False


def fuse_operators(op1, order1, op2):
    if merge_loops(order1, op2.output_order, op2.eval, op1):
        op2.compute.clear()
        op2.decl = [d for d in op2.decl if d.dobject != op2.eval]

def basic_rule(node, res):
    if type(node) == TensorOp and node.op_type in elementwise_op:
        if type(node.operators[0]) == TensorOp and len(node.operators[0].ref_by) == 1:
            if node.operators[0].op_type in (elementwise_op + ['setval', 'apply', 'einsum']):
                fuse_operators(node, node.input_orders[0], node.operators[0])

        if node.op_type in binary_elw:
            if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
                if node.operators[1].op_type in (elementwise_op + ['setval', 'apply', 'einsum']):
                   fuse_operators(node, node.input_orders[1], node.operators[1])


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
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
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
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
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
    f = fuser()
    f.register(basic_rule)
    code = codegen.cpu.print_cpp(f.fuse(ir1))
    print(code)


if __name__ == "__main__":
    test1()
    test2()
    test3()