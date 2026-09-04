from __future__ import annotations

from copy import deepcopy

from Tree.Tree import tree_node


def failure_node(parent, reason: str):
    """Create a converter-compatible terminal Thought for fail-closed episodes."""
    node = tree_node()
    node.node_type = "Thought"
    node.description = f"TaskGraph fail-closed: {reason}"
    node.io_state = deepcopy(parent.io_state)
    node.is_terminal = False
    node.messages = parent.messages.copy()
    node.father = parent
    node.observation_code = -1
    node.pruned = True
    parent.children.append(node)
    return node
