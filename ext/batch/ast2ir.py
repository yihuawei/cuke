from ext.batch.ast import *

def gen_ir(node):
    assert isinstance(node, Batch)
    if len(node.base.compute) > 0 or len(node.base.decl) > 0 or node.base.eval:
        return node
    if type(node) == BVec or type(node) == BMat or type(node) == BVar:
        node.base._gen_ir()
        node.batch_size._gen_ir()
        node.eval = node.base.eval

    return node

