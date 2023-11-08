import codegen.cpu
from core.ir import *
from core.asg import *


def rebind_iterate(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            rebind_iterate(l, old, new)
    elif type(ir) == Loop:
        rebind_iterate(ir.start, old, new)
        rebind_iterate(ir.end, old, new)
        rebind_iterate(ir.step, old, new)
        rebind_iterate(ir.body, old, new)
    elif type(ir) == Expr:
        rebind_iterate(ir.left, old, new)
        rebind_iterate(ir.right, old, new)
    elif type(ir) == Assignment:
        rebind_iterate(ir.lhs, old, new)
        rebind_iterate(ir.rhs, old, new)
    elif type(ir) == Ndarray:
        rebind_iterate(ir.size, old, new)
    elif type(ir) == Indexing:
        rebind_iterate(ir.dobject, old, new)
        if type(ir.idx) in (Scalar, Literal):
            if ir.idx == old:
                ir.idx = new
        else:
            rebind_iterate(ir.idx, old, new)
    elif type(ir) == Slice:
        rebind_iterate(ir.start, old, new)
        rebind_iterate(ir.stop, old, new)
        rebind_iterate(ir.step, old, new)
    elif type(ir) == Math:
        rebind_iterate(ir.val, old, new)

# def get_indices(ir, idx):
#     if type(ir) in (Scalar, Ndarray):
#         return ir
#     elif type(ir) is Indexing:
#         idx.append(ir.idx)
#         return get_indices(ir.dobject, idx)
#     else:
#         return None
#
# def find_assignments(ir, target, res):
#     if type(ir) == Assignment:
#         idx = []
#         item = get_indices(ir.lhs, idx)
#         if item != None and item == target:
#             res.append((ir, idx[::-1]))
#     elif type(ir) == Loop:
#         for s in ir.body:
#             find_assignments(s, target, res)
#
# def find_loop(loop, itr):
#     if type(loop) == Loop:
#         if loop.iterate == itr:
#             return loop
#         else:
#             for s in loop.body:
#                 r = find_loop(s, itr)
#                 if r != None:
#                     return r
#     return None

def gen_point_loops(loop_nest, outer_loops, tile_size, new_indices, num_tiled_loops, i):
    if i < len(loop_nest):
        l = loop_nest[i]
        if i < num_tiled_loops:
            tl = outer_loops[i]
            ts = tile_size[i]
            new_l = Loop(tl.iterate, Expr(Expr(tl.iterate, ts, '+'), l.end,'smaller'), l.step, [])
        else:
            new_l = Loop(l.start, l.end, l.step, [])
        outer_loops[-1].body.append(new_l)
        outer_loops.append(new_l)
        new_indices.append(new_l.iterate)
        new_l.body.append(gen_point_loops(loop_nest, outer_loops, tile_size, new_indices, num_tiled_loops, i+1))



def output_reorder(node, dim_order, tile_size):
    assert isinstance(node, TensorOp)
    assert node.op_type in arith_op or node.op_type in math_op or node.op_type in ('einsum', 'setval')
    assert len(node.compute) == 1
    assert len(dim_order) == len(tile_size) and len(dim_order) > 0
    loop = node.compute[0]

    for i in range(len(node._size())):
        # add the nontiled dimensions
        if not i in dim_order:
            dim_order.append(i)
            tile_size.append(0)

    assert sorted(dim_order) == list(range(len(dim_order)))

    loop_nest = []
    l = loop
    for i in range(len(dim_order)):
        loop_nest.append(l)
        l = loop.body[0]

    tile_loops = []
    reorder_nest = []
    new_indices = []
    output_order = []
    for i in range(len(dim_order)):
        l = loop_nest[dim_order[i]]
        if tile_size[i] > 0:
            tl = Loop(l.start, l.end, tile_size[i], [])
            if tile_size[i] == 1:
                new_indices.append(tl.iterate)
        else:
            tl = None
        tile_loops.append(tl)
        reorder_nest.append(l)
        output_order.append((dim_order[i], tl))

    for i in range(len(reorder_nest)):
        ts = tile_size[i]
        if ts > 1:
            tl = tile_loops[i]
            new_l = Loop(tl.iterate, Expr(Expr(tl.iterate, ts, '+'), l.end,'smaller'), l.step, [])
        elif ts == 1:
            new_l = None
        else:
            l = reorder_nest[i]
            new_l = Loop(l.start, l.end, l.step, [])
        if new_l != None:
            tile_loops.append(new_l)
            new_indices.append(new_l.iterate)

    tile_loops = [l for l in tile_loops if l != None]

    for i in range(len(tile_loops)-1, 0, -1):
        tile_loops[i-1].body.append(tile_loops[i])

    body = loop_nest[-1].body

    for i in range(len(dim_order)):
        old_itr = reorder_nest[i].iterate
        new_itr = new_indices[i]
        rebind_iterate(body, old_itr, new_itr)

    tile_loops[-1].body.extend(body)
    node.compute = [tile_loops[0]]
    node.output_order = output_order






if __name__ == "__main__":
    A = Tensor('a', (10, 20))
    B = Tensor('b', (20, 30))
    C = Tensor('c', (10, 30))
    res1 = A @ B
    ir1 = res1._gen_ir()
    print(res1.input_orders)
    print(res1.output_order)
    code = codegen.cpu.print_cpp(ir1)
    print(code)

    res2 = res1 + C
    ir2 = res2._gen_ir()
    print(res2.input_orders)
    print(res2.output_order)



