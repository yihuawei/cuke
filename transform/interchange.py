import codegen.cpu
from core.ir import *
from core.asg import *
from helpers import rebind_iterate


def interchange(node, new_order):
    assert isinstance(node, TensorOp)
    assert node.op_type in elementwise_op + ['apply', 'einsum', 'setval']





