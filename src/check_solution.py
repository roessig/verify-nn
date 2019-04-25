from model_boundd import MIPwithBounds
import networkx as nx

def get_vars_and_coefficients(elements, start=3):
    """Use a list which comes  from line.split() to create lists of float coefficients and SCIP variables."""
    return [var for var in elements[start + 1::2]], [float(coeff) for coeff in elements[start::2]]


def check_sol(filepath, value_dict, eps=1e-8, print_values=False):
    """Check solution given by input variables for feasibility.

    Args:
        filepath: str, path to .rlv file with AssertOut for output constraints
        value_dict: dict, mapping input variables names (str) to values of the solution
        eps: float, tolerance for checking

    Returns:
        true, if solution is valid, false otherwise
    """

    graph = nx.DiGraph()
    relu_nodes = set()
    max_pool_nodes = set()
    linear_nodes = set()
    relu_in_nodes = set()
    mip = MIPwithBounds(filepath, 1e-7)
    model, vars = mip.read_file_into_graph()
    # vars is a dict of the input nodes

    output_cons = []
    input_cons = []

    input_bounds = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elements = line.split()

            if elements[0] == "Input":
                input_bounds[elements[1]] = {"lb": None, "ub": None}
                graph.add_node(elements[1], node_type="input")

            if elements[0] == "ReLU":
                bias = float(elements[2])
                variables, coeffs = get_vars_and_coefficients(elements)

                relu_nodes.add(elements[1])
                graph.add_node(elements[1] + "_in", bias=bias)
                graph.add_edge(elements[1] + "_in", elements[1])
                relu_in_nodes.add(elements[1] + "_in")
                for v, w in zip(variables, coeffs):
                    graph.add_edge(v, elements[1] + "_in", weight=w)

            if elements[0] == "Linear":
                linear_nodes.add(elements[1])
                bias = float(elements[2])
                variables, coeffs = get_vars_and_coefficients(elements)

                graph.add_node(elements[1], bias=bias)
                for v, w in zip(variables, coeffs):
                    graph.add_edge(v, elements[1], weight=w)

            if elements[0] == "MaxPool":
                max_pool_nodes.add(elements[1])
                graph.add_node(elements[1], node_type="max_pool")
                graph.add_edges_from(((v, elements[1]) for v in elements[2:]), weight=1)

            if elements[0] == "AssertOut":
                output_cons.append((float(elements[2]), elements[1], get_vars_and_coefficients(elements)))

            if elements[0] == "Assert":
                input_cons.append((float(elements[2]), elements[1], get_vars_and_coefficients(elements)))
                """if len(elements) == 5 and elements[-1] in input_bounds:
                    if elements[1] == "<=":
                        new_lb = float(elements[2]) / float(elements[3])
                        if input_bounds[elements[-1]]["lb"] is None or input_bounds[elements[-1]]["lb"] < new_lb:
                            input_bounds[elements[-1]]["lb"] = new_lb

                    elif elements[1] == ">=":
                        new_ub = float(elements[2]) / float(elements[3])
                        if input_bounds[elements[-1]]["ub"] is None or input_bounds[elements[-1]]["ub"] > new_ub:
                            input_bounds[elements[-1]]["ub"] = new_ub"""




    val = True
    for lhs, direction, (variables, coeffs) in input_cons:
        if direction == "<=":
            if lhs > sum(c * value_dict[v] for v, c in zip(variables, coeffs)) + eps:
                val = False
                print(lhs, direction, variables, coeffs)
                break
        elif direction == ">=":
            if lhs < sum(c * value_dict[v] for v, c in zip(variables, coeffs)) - eps:
                val = False
                print(lhs, direction, variables, coeffs)

                break
        else:
            raise NotImplementedError

    if not val:  # input constraints do not hold
        print("input constraints not fulfilled")
        return False
    else:
        if print_values:
            print("input constraints hold")



    nodes_sorted = list(nx.topological_sort(graph))
    relu_phases = {x: -1 for x in relu_nodes}
    relu_phases_all = {x: 0 for x in relu_nodes}


    for node in nodes_sorted:
        if node in vars:
            continue   # skip the input nodes

        new_value = 0

        if node in linear_nodes or node in relu_in_nodes:
            for n in graph.predecessors(node):
                new_value += graph.edges[n, node]["weight"] * value_dict[n]

            new_value += graph.node[node]["bias"]

        elif node in max_pool_nodes:
            new_value = max(value_dict[n] for n in graph.predecessors(node))


        elif node in relu_nodes:
            pred = list(graph.predecessors(node))
            assert len(pred) == 1

            if value_dict[pred[0]] > 0:            # apply ReLU here
                new_value = value_dict[pred[0]]
                relu_phases[node] = 1
            else:
                relu_phases[node] = 0

        value_dict[node] = new_value


    for relu, phase in relu_phases.items():
        assert phase >= 0

        relu_phases_all[relu] += phase

    if print_values:
        for s in value_dict.items():
            print(s)

    val = True
    # check the ouput constraints
    #print(output_cons)
    for lhs, direction, (variables, coeffs) in output_cons:
        if direction == "<=":
            if lhs > sum(c * value_dict[v] for v, c in zip(variables, coeffs)) + eps:
                val = False
                break
        elif direction == ">=":
            if lhs < sum(c * value_dict[v] for v, c in zip(variables, coeffs)) - eps:
                val = False
                break
        else:
            raise NotImplementedError

    return val


if __name__ == "__main__":


    directory = "../benchmarks/collisionDetection/"
    directory2 = "../../benchmarks/scip/ACAS/"
    directory3 =  "../benchmarks/twinladder/"
    directory5_out = "../benchmarks/mnist/"
    filepath = directory2 + "property2/5_3.rlv"
    #filepath = directory2 + "property5/property.rlv"
    #filepath = directory2 + "property_3.rlv"

    file = "../logs/neurify_11_10_0_adv"
    with open(file, "r") as f:
        list_of_pixels = [float(x) for x in f.readline()[:-1].split()]

    #value_dict = {"in_" + str(i): x*255 for i, x in enumerate(list_of_pixels)}
    value_dict = {'in_0': 55947.69100, 'in_1': 0.198666, 'in_2': -3.051407, 'in_3': 1145.0000, 'in_4': 50.768384}



    if check_sol(filepath, value_dict=value_dict, eps=1e-2, print_values=True):
        print("valid solution found -> SAT")
    else:
        print("the solution is not valid")


