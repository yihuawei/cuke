from core.ir import *
from core.asg import *
from core.opt.reorder import rebind_iterate
import helpers

def _is_oob_indexing(ir, it, ofs):
    if type(ir) == Indexing:
        if (_is_oob_indexing(ir.dobject, it, ofs)) == True:
            return True
        if type(ir.idx) == Indexing and type(ir.idx.dobject) == Slice and ir.idx.idx.dobject_id == it.dobject_id and ('slice_ofs' in ir.idx.attr and ir.idx.attr['slice_ofs'] > ofs):
            return True
    return False

def _replace_oobindexing_with_padding(ir, it, ofs, val):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            _replace_oobindexing_with_padding(l, it, ofs, val)
    elif type(ir) == Loop:
        if _is_oob_indexing(ir.start, it, ofs):
            ir.start = val
        else:
            _replace_oobindexing_with_padding(ir.start, it, ofs, val)
        if _is_oob_indexing(ir.end, it, ofs):
            ir.end = val
        else:
            _replace_oobindexing_with_padding(ir.end, it, ofs, val)
        if _is_oob_indexing(ir.step, it, ofs):
            ir.step = val
        else:
            _replace_oobindexing_with_padding(ir.step, it, ofs, val)
        _replace_oobindexing_with_padding(ir.body, it, ofs, val)
    elif type(ir) == Expr:
        if _is_oob_indexing(ir.left, it, ofs):
            ir.left = val
        else:
            _replace_oobindexing_with_padding(ir.left, it, ofs, val)
        if _is_oob_indexing(ir.right, it, ofs):
            ir.right = val
        else:
            _replace_oobindexing_with_padding(ir.right, it, ofs, val)
    elif type(ir) == Assignment:
        if _is_oob_indexing(ir.rhs, it, ofs):
            ir.rhs = val
        else:
            _replace_oobindexing_with_padding(ir.rhs, it, ofs, val)
    elif type(ir) == Ndarray:
        _replace_oobindexing_with_padding(ir.size, it, ofs, val)
    elif type(ir) == Indexing:
        if _is_oob_indexing(ir.idx, it, ofs):
            ir.idx = val
        else:
            _replace_oobindexing_with_padding(ir.idx, it, ofs, val)
    elif type(ir) == Slice:
        if _is_oob_indexing(ir.start, it, ofs):
            ir.start = val
        else:
            _replace_oobindexing_with_padding(ir.start, it, ofs, val)
        if _is_oob_indexing(ir.stop, it, ofs):
            ir.stop = val
        else:
            _replace_oobindexing_with_padding(ir.stop, it, ofs, val)
        if _is_oob_indexing(ir.step, it, ofs):
            ir.step = val
        else:
            _replace_oobindexing_with_padding(ir.step, it, ofs, val)
    elif type(ir) == Math:
        if _is_oob_indexing(ir.val, it, ofs):
            ir.val = val
        else:
            _replace_oobindexing_with_padding(ir.val, it, ofs, val)

def _resolve_loops(scope):
    for stmt in scope[:]:
        if type(stmt) == Loop:
            _resolve_loops(stmt.body)
            if 'loop_ofs' in stmt.attr:
                for i in range(stmt.attr['loop_ofs']):
                    body = copy.deepcopy(stmt.body)
                    _replace_oobindexing_with_padding(body, stmt.iterate, i, Literal(0, 'int'))
                    rebind_iterate(body, stmt.iterate, Literal(i, 'int'))
                    idx = scope.index(stmt)
                    scope[idx:idx] = body
                stmt.start = Literal(stmt.attr['loop_ofs'], 'int')



def lower_bound_padding(asg):
    def action(node, res):
        if node.valid == True and type(node) == TensorOp and node.op_type in list(arith_op.keys()) + ['setval']:
            res.append(node.compute)

    t = helpers.Traversal(action)
    ir = t(asg)
    for l in ir:
        _resolve_loops(l)

