import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import examples
from cset.app import subgraph_matching
from cset.app import decomine

if __name__ == "__main__":
    subgraph_matching.SubgraphMatchingEdgeInduced(sys.argv[1])
    # decomine.p3_edge_induced2()