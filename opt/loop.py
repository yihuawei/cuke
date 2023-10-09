from core.ir import *

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
    # elif type(ir) == Ref:
    #     rebind_iterate(ir.dobject, old, new)
    elif type(ir) == Index:
        rebind_iterate(ir.dobject, old, new)
        rebind_iterate(ir.ind_arr, old, new)
        if type(ir.index) == Scalar:
            if ir.index == old:
                ir.index = new
        else:
            rebind_iterate(ir.index, old, new)
    elif type(ir) == Slice:
        rebind_iterate(ir.start, old, new)
        rebind_iterate(ir.stop, old, new)
        rebind_iterate(ir.step, old, new)

def fuse(scope, loop1, loop2):
    if type(loop1) == Loop and type(loop2) == Loop:
        if loop1.start == loop2.start and loop1.end == loop2.end and loop1.step == loop2.step:
            rebind_iterate(loop2.body, loop2.iterate, loop1.iterate)
            loop1.body.extend(loop2.body[:])
            scope.remove(loop2)
