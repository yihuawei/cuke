import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import examples
from cset import examples as cset_examples

if __name__ == "__main__":
    cset_examples.subgraph_matching(sys.argv[1])