from collections import deque

def bfs_dist(graph, source):
    """Copied partly from networkx.

    Args:
        graph: instance of networkx.Graph
        source: a node of graph

    Returns:
        dict of distances from source for all nodes, measured in number of edges
    """

    visited = set([source])
    queue = deque([(source, graph.neighbors(source))])

    distances = {source: 0}
    while queue:
        parent, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                visited.add(child)
                queue.append((child, graph.neighbors(child)))
                distances[child] = distances[parent] + 1
        except StopIteration:
            queue.popleft()

    return distances