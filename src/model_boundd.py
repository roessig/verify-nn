from pyscipopt import Model, quicksum, SCIP_PROPTIMING, SCIP_PRESOLTIMING, SCIP_EVENTTYPE, SCIP_HEURTIMING
import networkx as nx
from collections import OrderedDict
from relu_branching import ReluBranching
from domain_branching import DomainBranching
from sampling_heuristic import SamplingHeuristic
from utils import bfs_dist
from lb_event import LbChangeEvent
from dualbound_event import DualBoundEvent
import torch.nn
from compute_bounds import BoundComp
from collections import namedtuple
from relu_sepa import ReluSepa


class MIPwithBounds:
    """Base class of our solver implementation. Contains most data structures and is used to include the
    different components such as separators, branching rules, propagators. """
    
    def __init__(self, filepath, eps):
        self.model = Model("")
        self.vars = {}
        self.eps = eps
        self.filepath = filepath
        self.graph = nx.DiGraph()
        self.relu_nodes = {}
        self.relu_in_nodes = {}
        self.fixed_positive = {}
        self.fixed_negative = {}
        self.relu_cons = {}
        self.max_pool_nodes = {}
        self.linear_nodes = {}
        self.input_nodes = {}
        self.binary_variables = {}
        self.nodes_sorted = []
        self.layers = []   # list of sets, each set represents one layer, sets contain variable names (str)
        self.node_position_pytorch = {}   # maps neuron names to layer number and index in layer in the pytorch model
                                            # values are tuples of ints (layer, <neuron in layer>)
        self.delete_cons = []
        self.output_cons = []
        self.output_variables = {}   # dict of those variable names, which are introduced for the optimization mode,
                                     # except self.objective_variable
        self.output_variables_binary = []   # list of the binary vars for the output, these vars are ALSO in self.binary_vars
        self.verify_or_constraints = False
        self.objective_variable = None
        self.nodes_by_branch_prio = {}   # dict {scip_node_number: possible branch Relus}
        self.pytorch_model = None
        self.dualbound_hdlr_feas = None
        self.dualbound_hdlr_infeas = None
        self.local_search_hdlr = None
        self.debug_bound_hdlr = None
        self.bound_comp = None


    def add_cons_relu_linear(self, x, y):
        """Add linear approximation of a ReLU constraint as in Ehlers (2017).

        Args:
            x: input variable
            y: output variabe, lower bound of this variable should be 0
            lb: float, lower bound
            ub: float, upper bound
        """

        self.model.addCons(y >= x)
        ub = x.getUbGlobal()
        lb = x.getLbGlobal()
        factor = ub / (ub - lb)
        if lb <= 0 and ub >= 0:
            self.delete_cons.append(self.model.addCons(y + lb * factor <= x * factor))


    def add_cons_maxpool_linear(self, X, y, lbs, name):
        """Add linear approximation of a ReLU constraint as in Ehlers (2017).

        Args:
            X: list of input variables
            y: output variable, lower bound should be -inf
            lbs: list of float, lower bounds of the input variables
        """

        self.delete_cons.append(self.model.addCons(quicksum(_x for _x in X) >= y + sum(lbs) - max(lbs),
                                                   name=name + "_lin_approx"))
        for _x in X:
            self.model.addCons(y >= _x, name=name + "_lb")

    
    
    def add_cons_maxpool(self, X, y, M, name, use_bound_disj=False, use_mip=True):
        """Add a max pool constraint to self.model, i.e. constraint y = max(X)
    
        Args:
            self.model: instance of pyscipopt self.model
            X: list of input variables
            y: output variable, lower bound should be -inf (unless all inputs are >= 0)
            M: upper bound on all input variables
        """
    
        if use_bound_disj:
            num = len(X)
            a = [self.model.addVar(lb=None) for _ in range(num)]
            for _a, _x in zip(a, X):
                self.model.addCons(_a == _x - y)
                self.model.addCons(y >= _x)
    
            self.model.addConsBoundDisjunction(a, ["lb" for _ in range(num)], [0 for _ in range(num)])

        if use_mip:
            assert name
            print("add maxpool", X, y, M)
            d_vars = []
            for i, _x in enumerate(X):
                self.model.addCons(y >= _x, name=name + "_lb_" + _x.name)
                _d = self.model.addVar(ub=1, vtype="B", name="bin_" + name + "_" + _x.name)
                self.model.addCons(y <= _x + (M - _x.getLbGlobal()) * (1 - _d), name=name + "_bin_ub_" + _x.name)
                d_vars.append(_d)
                self.binary_variables["bin_" + name + "_" + _x.name] = _d
                self.output_variables_binary.append(_d)

            self.model.addCons(1 == quicksum(_d for _d in d_vars), name=name + "_sum")



    def compute_linear_bounds(self, variables, coefficients, local=False):
        """Copmute the upper and lower bound before the ReLU application to the given variables with the
        corresponding coefficients. If lower bound >= 0 or upper bound <= 0, then the phase of the ReLU
        can be fixed.

        Args:
            variables: list of str, containing the names of the variables
            coefficients: list of float coefficients
            local: bool, if True local bounds are used, otherwise global bounds

        Returns:
            tuple: (lower bound, upper bound)
        """

        assert len(variables) == len(coefficients)

        lb, ub = 0, 0
        for v, c in zip(variables, coefficients):
            v = self.vars[v]
            if local:
                current_lb = v.getLbLocal()
                current_ub = v.getUbLocal()
            else:
                current_lb = v.getLbGlobal()
                current_ub = v.getUbGlobal()

            if c > 0:
                ub += current_ub * c
                lb += current_lb * c
            elif c < 0:
                ub += current_lb * c
                lb += current_ub * c


        return lb, ub


    def quicksum_coeff_var(self, elements):
        return quicksum(c * v for v, c in zip(*self.get_vars_and_coefficients(elements)))

    def quicksum_from_var_names(self, variable_names, coeffs):
        return quicksum(c * self.vars[v] for v, c in zip(*(variable_names, coeffs)))

    def get_vars_and_coefficients(self, elements, start=3, str_only=False):
        """Use a list which comes  from line.split() to create lists of float coefficients and SCIP variables."""
        if str_only:
            return [var for var in elements[start + 1::2]], [float(coeff) for coeff in elements[start::2]]
        else:
            return [self.vars[var] for var in elements[start + 1::2]], [float(coeff) for coeff in elements[start::2]]


    def read_file_into_graph(self):
        """Read the input file and add all neurons to self.graph. Futhermore, the input constraints are added
        to self.model and self.output_cons is filled."""

        assert_or = False
        assert_and = False
        input_bounds = {}
        input_elements = []

        with open(self.filepath, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                elements = line.split()
                if elements[0] == "Input":
                    input_bounds[elements[1]] = {"lb": None, "ub": None}
                    self.graph.add_node(elements[1], node_type="input")

                if elements[0] == "ReLU":

                    relu_in_name = elements[1] + "_in"
                    relu_out_name = elements[1]
                    bias = float(elements[2])
                    variables, coeffs = self.get_vars_and_coefficients(elements, str_only=True)

                    relu_entry = namedtuple("ReLU", ["relu_in_name", "relu_out_name", "bias", "variables", "coeffs"])
                    self.relu_nodes[elements[1]] = relu_entry(relu_in_name, relu_out_name, bias, variables, coeffs)
                    # relu_in node is created and added to self.relu_in_nodes later, other variables also created later

                    self.graph.add_node(relu_in_name, node_type="relu_in", bias=bias)
                    self.graph.add_node(relu_out_name, node_type="relu_out")
                    self.graph.add_edge(relu_in_name, relu_out_name)
                    for v, w in zip(variables, coeffs):
                        self.graph.add_edge(v, relu_in_name, weight=w)


                if elements[0] == "MaxPool":

                    self.max_pool_nodes[elements[1]] = elements[2:]
                    self.graph.add_node(elements[1], node_type="max_pool")
                    self.graph.add_edges_from(((v, elements[1]) for v in elements[2:]), weight=1)


                if elements[0] == "Linear":
                    variables, coeffs = self.get_vars_and_coefficients(elements, str_only=True)
                    bias = float(elements[2])
                    self.linear_nodes[elements[1]] = (bias, variables, coeffs)

                    #self.graph.add_edges_from((v.name, linear.name) for v in variables)
                    self.graph.add_node(elements[1], node_type="linear", bias=bias)
                    for v, w in zip(variables, coeffs):
                        self.graph.add_edge(v, elements[1], weight=w)


                if elements[0] == "Assert":
                    input_elements.append(elements)

                    # explicit bounds for input neurons
                    if len(elements) == 5 and elements[-1] in input_bounds:
                        if elements[1] == "<=":
                            new_lb = float(elements[2]) / float(elements[3])
                            if input_bounds[elements[-1]]["lb"] is None or input_bounds[elements[-1]]["lb"] < new_lb:
                                input_bounds[elements[-1]]["lb"] = new_lb

                        elif elements[1] == ">=":
                            new_ub = float(elements[2]) / float(elements[3])
                            if input_bounds[elements[-1]]["ub"] is None or input_bounds[elements[-1]]["ub"] > new_ub:
                                input_bounds[elements[-1]]["ub"] = new_ub

                if elements[0] == "AssertOut":
                    assert elements[1] in ["<=", ">="] and not assert_or
                    assert_and = True
                    cons = namedtuple("output_cons", ["lhs", "operator", "elements"])
                    self.output_cons.append(cons(float(elements[2]), True if elements[1] == ">=" else False, elements))
                    #print("assertout", elements)

                if elements[0] == "AssertOr":
                    # assertOr properties are basically the same as AND, just the "imaginary" operator is turned around
                    # we assume that all subsequent ORs in the file form one disjunction
                    assert elements[1] in ["<=", ">="] and not assert_and
                    assert_or = True
                    cons = namedtuple("output_cons", ["lhs", "operator", "elements"])
                    self.output_cons.append(cons(float(elements[2]), False if elements[1] == ">=" else True, elements))

        for var_name, bounds in input_bounds.items():
            self.vars[var_name] = self.model.addVar(name=var_name, lb=bounds["lb"], ub=bounds["ub"])
            self.input_nodes[var_name] = self.vars[var_name]

        for assert_input_count, elements in enumerate(input_elements):
            if elements[1] == "<=":
                self.model.addCons(float(elements[2]) <= self.quicksum_coeff_var(elements),
                                   name="input_cons_" + str(assert_input_count))
            elif elements[1] == ">=":
                self.model.addCons(float(elements[2]) >= self.quicksum_coeff_var(elements),
                                   name="input_cons_" + str(assert_input_count))
            else:
                raise NotImplementedError("This property cannot be verified: " + elements[1])

        self.model.hideOutput()
        for var_name, bounds in input_bounds.items():
            if bounds["lb"] is None:
                self.model.setObjective(self.vars[var_name])
                self.model.optimize()
                if self.model.getStatus() != "optimal":
                    raise ValueError("LP lower bound of input cannot be solved to optimality")
                else:
                    bound = self.model.getDualbound()
                    self.model.freeTransform()
                    self.model.chgVarLbGlobal(self.vars[var_name], bound - 10 * self.eps)
            if bounds["ub"] is None:
                self.model.setObjective(self.vars[var_name], sense="maximize")
                self.model.optimize()
                if self.model.getStatus() != "optimal":
                    raise ValueError("LP upper bound of input cannot be solved to optimality")
                else:
                    bound = self.model.getDualbound()
                    self.model.freeTransform()
                    self.model.chgVarUbGlobal(self.vars[var_name], bound + 10 * self.eps)

            self.model.setObjective(0.0)

        self.model.hideOutput(quiet=False)

        return self.model, self.vars


    def add_further_constraints(self, linear_model=False, optimize_nodes=False, opt_mode=False, use_symbolic=False,
                                bfs_from_all_inputs=False):
        """This methods adds all variables and also fills self.nodes_sorted

        Args:
            linear_model:        bool, if True add only linear approximation constraints instead of "real" constraints
            optimize_nodes:      bool, should nodes be optimized (implies solving MIPs/LPs depending on linear_model=False/True)
            opt_mode:            bool, use the optimization mode of Bunel et al. or not?
            use_symbolic:        bool, use the symbolic bound computation of Wang et al. ?
            bfs_from_all_inputs: bool, should a Breadth first search be performed from all inputs rather than just one?
                                    This is needed, if some input neurons are not connected to all neurons in the
                                    next layer, or there are layers which are not fully connected
        """

        self.bound_comp = BoundComp(self)   # must be called after self.build_model() to have input nodes available

        if optimize_nodes:
            self.model.setRealParam("limits/time", 3)
            self.model.hideOutput()

        self.nodes_sorted = list(nx.topological_sort(self.graph))

        if bfs_from_all_inputs: # this is required if not all neurons in the NN can be reached from every input neuron
            for input_node in self.input_nodes:
                for node, dist in bfs_dist(self.graph, input_node).items():
                    if dist >= len(self.layers):
                        self.layers.append(set())
                    self.layers[dist].add(node)
        else:
            for node, dist in bfs_dist(self.graph, next(iter(self.input_nodes))).items():
                if dist >= len(self.layers):
                    self.layers.append(set())
                self.layers[dist].add(node)
            self.layers[0].update(self.input_nodes)

        for i, s in enumerate(self.layers):
            self.layers[i] = sorted(s)

        num_fixed = 0
        for layer_index, layer_nodes in enumerate(self.layers):

            if use_symbolic:
                temp_values = self.bound_comp.update_symbolic_bounds(layer_nodes, use_approximation=True,
                                                                compare_with_global=False)

            for node_name in layer_nodes:

                if node_name in self.relu_nodes:
                    # this is filled in dnn_bound_prop, the dict will map node_number to the constraints
                    self.relu_cons[node_name] = {}

                    relu_in_name, relu_out_name, bias, variable_names, coeffs = self.relu_nodes[node_name]
                    lb, ub = self.compute_linear_bounds(variable_names, coeffs)
                    lb += bias
                    ub += bias

                    relu_out = self.model.addVar(name=relu_out_name)  # output of ReLU with lb=0
                    relu_in = self.model.addVar(name=relu_in_name, lb=lb, ub=ub)

                    self.vars[node_name] = relu_out
                    self.vars[node_name + "_in"] = relu_in
                    self.relu_in_nodes[node_name + "_in"] = relu_in


                    self.model.addCons(relu_in == bias + self.quicksum_from_var_names(variable_names, coeffs),
                                       name=node_name + "_in")

                    if optimize_nodes and abs(ub) + abs(lb) < 200:
                        self.model.setObjective(relu_in)
                        self.model.optimize()
                        stat = self.model.getStatus()
                        new_lb = self.model.getDualbound()

                        self.model.freeTransform()

                        self.model.setObjective(relu_in, sense="maximize")
                        self.model.optimize()
                        new_ub = self.model.getDualbound()

                        self.model.freeTransform()



                    elif use_symbolic:
                        new_lb = max(lb, temp_values[node_name][0])
                        new_ub = min(ub, temp_values[node_name][1])

                    else:
                        new_lb = lb
                        new_ub = ub


                    new_lb -= self.eps
                    new_ub += self.eps

                    self.model.tightenVarLbGlobal(relu_in, new_lb)
                    self.model.tightenVarUbGlobal(relu_in, new_ub)

                    # add the ReLU (approximation or binary) constraint for newly added node to the model
                    if new_lb < 0 < new_ub:
                        self.model.tightenVarUbGlobal(relu_out, new_ub)
                        self.model.addCons(relu_out >= relu_in, name=node_name + "_relu_lb")

                        if linear_model:
                            factor = new_ub / (new_ub - new_lb)
                            self.delete_cons.append(self.model.addCons(relu_out + new_lb * factor <= relu_in * factor,
                                                                       name=node_name + "_lin_approx"))
                        else:

                            d = self.model.addVar(ub=1, vtype="B", name="bin_" + node_name)
                            self.binary_variables["bin_" + node_name] = d
                            c1 = self.model.addCons(relu_out <=
                                                    relu_in - (1 - d) * (new_lb + self.eps), name=node_name + "_bin_lb")
                            c2 = self.model.addCons(relu_out <= d * (new_ub + self.eps), name=node_name + "_bin_ub")

                    elif new_ub <= 0:
                        num_fixed += 1
                        self.model.fixVar(relu_out, 0)
                        self.fixed_negative[node_name] = relu_out

                    elif new_lb >= 0:
                        num_fixed += 1
                        self.model.addCons(relu_out == relu_in, name=node_name + "_fix_pos")
                        self.model.tightenVarLbGlobal(relu_out, new_lb)
                        self.model.tightenVarUbGlobal(relu_out, new_ub)
                        self.fixed_positive[node_name] = relu_out

                elif node_name in self.linear_nodes:
                    bias, variable_names, coeffs = self.linear_nodes[node_name]
                    lb, ub = self.compute_linear_bounds(variable_names, coeffs)

                    linear = self.model.addVar(name=node_name, lb=lb+bias, ub=ub+bias)
                    self.model.addCons(linear == bias + self.quicksum_from_var_names(variable_names, coeffs),
                                       name=node_name + "_linear")
                    self.vars[node_name] = linear

                elif node_name in self.max_pool_nodes:
                    max_pool = self.model.addVar(name=node_name, lb=None)
                    variables = [self.vars[var_name] for var_name in self.max_pool_nodes[node_name]]
                    self.vars[node_name] = max_pool
                    M = max(v.getUbGlobal() for v in variables)
                    self.model.chgVarUbGlobal(max_pool, M)
                    self.model.chgVarLbGlobal(max_pool, min(v.getLbGlobal() for v in variables))
                    if linear_model:
                        self.add_cons_maxpool_linear(variables, max_pool, [v.getLbGlobal() for v in variables], node_name)
                    else:
                        self.add_cons_maxpool(variables, max_pool, M, name=node_name)

            # end of current layer
            # here we assume that all ReLU layers contain only RelU nodes

        print(num_fixed, "variables fixed")


        if optimize_nodes:
            self.model.setObjective(0.0)
            self.model.hideOutput(quiet=False)

        # currently AssertOr is only supported if opt_mode = True
        if not opt_mode:
            for lhs, operator, elements in self.output_cons:
                assert elements[0] == "AssertOut", "AssertOr only supported in opt_mode"
                if not operator:
                    self.model.addCons(lhs <= self.quicksum_coeff_var(elements))
                else:
                    self.model.addCons(lhs >= self.quicksum_coeff_var(elements))



    def _build_output_cons(self, lhs, operator, elements, name, opt_mode):
        """Build an output constraint and add it to self.graph. Also added to the model if opt_mode is True.

        Args:
            lhs:        float, left hand side of the constraint
            operator:   bool, indicates the direction of the operator
            elements:   list, elements as saved in self.output_cons
            name:       str, name that constraint shall have
            opt_mode:   bool, is opt_mode used?

        Returns:
            float, float -- the lower and upper bound of the constraint neuron (to be used if opt_mode == True)
        """

        var_name = name
        variable_names, coeffs = self.get_vars_and_coefficients(elements, str_only=True)
        self.output_variables[var_name] = variable_names, coeffs, operator, lhs
        if opt_mode:
            out_var = self.model.addVar(var_name, lb=None, ub=None)
            self.vars[var_name] = out_var

        variables = [self.vars[v] for v in variable_names]
        lb, ub = self.compute_linear_bounds(variable_names, coeffs)

        if operator:
            if opt_mode:
                self.model.addCons(out_var == self.quicksum_coeff_var(elements) - lhs, name=var_name)
                self.model.chgVarUbGlobal(out_var, ub - lhs)
                self.model.chgVarLbGlobal(out_var, lb - lhs)
            ub_return = ub - lhs
            lb_return = lb - lhs
            self.graph.add_node(var_name, node_type="linear_opt", bias=-lhs)
            for v, w in zip(variables, coeffs):
                self.graph.add_edge(v.name, var_name, weight=w)

        else:
            if opt_mode:
                self.model.addCons(out_var == lhs - self.quicksum_coeff_var(elements), name=var_name)
                self.model.chgVarUbGlobal(out_var, lhs - lb)
                self.model.chgVarLbGlobal(out_var, lhs - ub)
            ub_return = lhs - lb
            lb_return = lhs - ub
            # in this case we have to switch the sign of all coefficients
            self.graph.add_node(var_name, node_type="linear_opt", bias=lhs)
            for v, w in zip(variables, coeffs):
                self.graph.add_edge(v.name, var_name, weight=-w)

        return lb_return, ub_return


    def add_optimize_constraints(self, opt_mode):
        """We always need this function to build the pytorch model correctly. Only if opt_mode == True,
        the optimize cons are added to the SCIP model.
        Add constraints as in PLNN paper to model verification as optimization problem. Needs that the dict
        self.output_cons is filled correctly with all output constraints.

        Notice that in the rlv-Files the AssertOut lines are not the properties, but the inverse properties. In the
        PLNN paper, they talk about the actual (not inverse) properties. If the original properties are conjunctions,
        then the inverted properties are disjunctions.

        Args:
            opt_mode: bool, should the opt constraints be added to the SCIP model?
        """

        upper_bounds = []
        lower_bounds = []

        for i, (lhs, operator, elements) in enumerate(self.output_cons):
            if elements[0] == "AssertOr":
                assert opt_mode, "AssertOr only allowed in opt_mode"
                self.verify_or_constraints = True
                lb, ub = self._build_output_cons(lhs, operator, elements, "output_cons_or_" + str(i), opt_mode)
                lower_bounds.append(lb)
                upper_bounds.append(ub)

            elif elements[0] == "AssertOut":
                lb, ub = self._build_output_cons(lhs, operator, elements, "output_cons_" + str(i), opt_mode)
                lower_bounds.append(lb)
                upper_bounds.append(ub)

            else:
                raise TypeError("Error with AssertOut and Or")

        self.graph.add_node("t", node_type="max_pool_opt")
        self.graph.add_edges_from(((v, "t") for v in self.output_variables), weight=1)

        # add max pool "node" that enforces all output constraints
        if opt_mode:
            self.objective_variable = self.model.addVar(name="t", lb=None)
            self.model.chgVarUbGlobal(self.objective_variable, max(upper_bounds))
            self.model.chgVarLbGlobal(self.objective_variable, max(lower_bounds))

            self.add_cons_maxpool([self.vars[v] for v in self.output_variables], self.objective_variable,
                                      max(upper_bounds), name="max_pool_opt")

            # if we verify or constraints, we switch the optimization direction since there is a minus sign that
            # must be simulated
            self.model.setObjective(-self.objective_variable if self.verify_or_constraints else self.objective_variable)


    def add_binary_constraints(self, delete_cons=True):
        """Add Relu and max pool constraints after using the function optimize_bounds_lp.
        relu_out >= relu_in constraints are not added in this function, and should therefore not
        be contained in self.delete_cons. Also adds the maxpool constraint for the optimization
        approach, if self.objective variable exists.

        Args:
            delete_cons: bool, if True, the linear approximation constraints are deleted,
            otherwise they remain in the problem formulation
        """

        if delete_cons:
            for cons in self.delete_cons:
                self.model.delCons(cons)

        for node_name in self.nodes_sorted:

            if node_name in self.relu_nodes and \
                    node_name not in self.fixed_positive and node_name not in self.fixed_negative:
                relu_in, relu_out, bias, variable_names, coeffs = self.relu_nodes[node_name]
                relu_in = self.vars[relu_in]
                relu_out = self.vars[relu_out]

                lb = relu_in.getLbGlobal()
                ub = relu_in.getUbGlobal()
                d = self.model.addVar(ub=1, vtype="B", name="bin_" + node_name)
                self.binary_variables["bin_" + node_name] = d
                c1 = self.model.addCons(relu_out <= relu_in - (1 - d) * (lb - self.eps), name=node_name + "_bin_lb")
                c2 = self.model.addCons(relu_out <= d * (ub + self.eps), name=node_name + "_bin_ub")
                #self.relu_cons[node_name] = c1, c2

                if lb >= 0:
                    self.model.chgVarLbGlobal(self.binary_variables["bin_" + node_name], 1)
                if ub <= 0:
                    self.model.chgVarUbGlobal(self.binary_variables["bin_" + node_name], 0)

            elif node_name in self.max_pool_nodes:
                variables = [self.vars[var] for var in self.max_pool_nodes[node_name]]

                M = max(v.getUbGlobal() for v in variables)
                self.add_cons_maxpool(variables, self.vars[node_name], M, name=node_name)



    def build_pytorch_model(self):
        """Builds a neural net class for pytorch use. Can only be called after self.graph was created
        in add_further_constraints().
        """

        layer_type_list = []
        for i, layer_nodes in enumerate(self.layers):

            layer_size = len(layer_nodes)
            layer_sample_node_type = self.graph.node[next(iter(layer_nodes))]["node_type"]
            if layer_sample_node_type in ["linear", "relu_out", "max_pool"]:
                layer_type_list.append((layer_sample_node_type, layer_size, i))
                for j, n in enumerate(layer_nodes):
                    self.node_position_pytorch[n] = (len(layer_type_list) - 1, j)
            elif layer_sample_node_type in ["relu_in", "input"]:
                pass
            else:
                raise TypeError("Layer type not correct or not supported, was " + layer_sample_node_type)

        class NNClass(torch.nn.Module):
            """Class for the  pytorch model.
            Args:
                num_inputs: int, number of input neurons

            """
            def __init__(self, num_inputs, num_output_cons=None, opt_mode=False):
                """
                Args:
                    num_inputs: int, number of input neurons
                    num_output_cons: int, number of output constraints, is only used in opt mode
                    opt_mode: bool, should the pytorch network be constructed including the opt neurons
                """

                super(NNClass, self).__init__()
                last_size = num_inputs
                self.opt_mode = opt_mode

                for i, (layer_type, layer_size, _) in enumerate(layer_type_list):
                    if layer_type in ["linear", "relu_out"]:
                        setattr(self, str(i) + "_" + layer_type, torch.nn.Linear(last_size, layer_size))
                    elif layer_type == "max_pool":
                        assert layer_size == 1, "More than one max pool node per layer currently not supported, see TODO"
                        setattr(self, str(i) + "_" + layer_type, torch.nn.MaxPool1d(last_size))    # last_size = kernel_size
                    else:
                        raise TypeError("Wrong layer type.")
                    last_size = layer_size

                # we always add the "optimization neurons" to the pytorch model since these help to
                # find primal solutions using the heuristic
                assert num_output_cons > 0, "Number of outputs must be given in order to use opt mode."
                setattr(self, str(len(layer_type_list)) + "_linear_opt", torch.nn.Linear(last_size, num_output_cons))
                setattr(self, str(len(layer_type_list) + 1) + "_max_pool_opt", torch.nn.MaxPool1d(num_output_cons))

            def forward(self, x):

                for i, (layer_type, layer_size, _) in enumerate(layer_type_list):
                    if layer_type == "max_pool":
                        x = x.view(1, 1, -1)
                    x = getattr(self, str(i) + "_" + layer_type)(x)
                    if layer_type == "relu_out":
                        x = torch.nn.functional.relu(x)

                # we always add the "optimization neurons" to the pytorch model since these help to
                # find primal solutions using the heuristic
                x = getattr(self, str(len(layer_type_list)) + "_linear_opt")(x)
                x = x.view(1, 1, -1)
                x = getattr(self, str(len(layer_type_list) + 1) + "_max_pool_opt")(x)
                return x

        self.pytorch_model = NNClass(len(self.input_nodes), len(self.output_variables), True)

        print(self.pytorch_model.state_dict().keys())
        state_dict = OrderedDict()

        # here we only support neuron predecessors from the layer immediately before
        # would require some substantial changes to the pytorch access to change this
        for torch_index, (layer_type, layer_size, orig_index) in enumerate(layer_type_list):
            if layer_type == "max_pool":
                continue    # for max pool layers we don't have to create a pytorch tensor
            layer_weights = []
            layer_biases = []
            for el in self.layers[orig_index]:   # + 1 to skip input layer
                pred_weights = []
                if layer_type == "relu_out":
                    predecessors = set(self.graph.predecessors(el + "_in"))
                    for possible_pred in self.layers[orig_index - 2]:   # skip the relu_in layer
                        if possible_pred in predecessors:
                            predecessors.remove(possible_pred)
                            pred_weights.append(self.graph.edges[possible_pred, el + "_in"]["weight"])
                        else:
                            pred_weights.append(0)
                    layer_biases.append(self.graph.node[el + "_in"]["bias"])


                elif layer_type == "linear":
                    predecessors = set(self.graph.predecessors(el))
                    for possible_pred in self.layers[orig_index - 1]:
                        if possible_pred in predecessors:
                            predecessors.remove(possible_pred)
                            pred_weights.append(self.graph.edges[possible_pred, el]["weight"])
                        else:
                            pred_weights.append(0)
                    layer_biases.append(self.graph.node[el]["bias"])

                layer_weights.append(pred_weights)
                assert len(predecessors) == 0, "set of predecessors not empty, i.e. predecessor from previous layer"

            state_dict[str(torch_index) + "_" + layer_type + ".weight"] = torch.FloatTensor(layer_weights)
            state_dict[str(torch_index) + "_" + layer_type + ".bias"] = torch.FloatTensor(layer_biases)

        if True:  # use_opt_mode
            layer_weights = []
            layer_biases = []
            for el in self.output_variables:
                pred_weights = []
                predecessors = set(self.graph.predecessors(el))
                for possible_pred in self.layers[-1]:           # works only with constraints on output variables,
                                                                # not on other variables
                    if possible_pred in predecessors:
                        predecessors.remove(possible_pred)
                        pred_weights.append(self.graph.edges[possible_pred, el]["weight"])
                    else:
                        pred_weights.append(0)
                layer_biases.append(self.graph.node[el]["bias"])
                layer_weights.append(pred_weights)

            state_dict[str(len(layer_type_list)) + "_linear_opt.weight"] = torch.FloatTensor(layer_weights)
            state_dict[str(len(layer_type_list)) + "_linear_opt.bias"] = torch.FloatTensor(layer_biases)

        self.pytorch_model.load_state_dict(state_dict)
        self.pytorch_model.eval()


    def add_relu_branching(self, **kwargs):
        self.model.includeBranchrule(ReluBranching(self),
            "relu_branch_rule", "branch on ReLU nodes", **kwargs)

    def add_domain_branching(self, opt_mode, split_mode, **kwargs):
        self.model.includeBranchrule(DomainBranching(self, opt_mode, split_mode),
            "domain_branch_rule", "branch input domains", **kwargs)

    def add_sampling_heuristic(self, freq, maxdepth, **kwargs):
        self.model.includeHeur(SamplingHeuristic(self, **kwargs),
                               "sampling_heuristic", "try random solutions", "s", freq=freq, maxdepth=maxdepth)

    def add_sampling_heuristic_local(self, freq, maxdepth, **kwargs):
        self.model.includeHeur(SamplingHeuristic(self, use_local_bounds=True, **kwargs),
                               "sampling_heuristic_local", "try random locally", "l",
                               priority=100000, freq=freq, maxdepth=maxdepth)

    def add_relu_sepa(self, priority=10, freq=1, maxbounddist=1.0, delay=False):
        self.model.includeSepa(ReluSepa(self), "relu_sepa", "ideal separation", priority=priority, freq=freq,
                               maxbounddist=maxbounddist, delay=delay)

    def add_dnn_bound_prop(self, opt_mode, optimize_nodes, obbt_2, use_symbolic, bound_for_opt, maxdepth, use_genvbounds, **kwargs):
        if opt_mode:
            from dnn_bound_prop_opt import DNNBoundProp
        else:
            from dnn_bound_prop_nonopt import DNNBoundProp
        prop = DNNBoundProp(self, optimize_nodes, obbt_2, use_symbolic, bound_for_opt, maxdepth, use_genvbounds, **kwargs)
        self.model.includeProp(prop,
                               "use lp approximation", "bound tightening for domain branching",
                               presolpriority=1000,
                               presolmaxrounds=0, proptiming=SCIP_PROPTIMING.AFTERLPLOOP,
                               priority=9999999, freq=1, delay=True,
                               presoltiming=SCIP_PRESOLTIMING.EXHAUSTIVE)
        return prop

    def add_eventhdlr_debug(self):
        """Detect bound change events for debugging purposes.
        Put the variable in question as parameter to .getTransformedVar. SCIP_EVENTTYPE can be set to
        UBTIGHTENED or LBTIGHTENED"""
        self.debug_bound_hdlr = LbChangeEvent()
        self.model.includeEventhdlr(self.debug_bound_hdlr, "ub change", "t")


    def add_eventhdlr_dualbound(self):
        self.dualbound_hdlr_feas = DualBoundEvent(self)
        self.model.includeEventhdlr(self.dualbound_hdlr_feas, "dualbound_feas", "check dualbound value")
        self.dualbound_hdlr_infeas = DualBoundEvent(self)
        self.model.includeEventhdlr(self.dualbound_hdlr_infeas, "dualbound_infeas", "check dualbound value")

    def catch_events(self):
        """Must be called to activate event handlers. Not necessary for eventhdlr_debug."""
        if self.dualbound_hdlr_feas is not None and self.dualbound_hdlr_infeas is not None:
            self.model.catchEvent(SCIP_EVENTTYPE.NODEINFEASIBLE, self.dualbound_hdlr_infeas)
            self.model.catchEvent(SCIP_EVENTTYPE.NODEFEASIBLE, self.dualbound_hdlr_feas)
            print("caught event handler dualbound")
        if self.local_search_hdlr is not None:
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self.local_search_hdlr)
            print("caught ")
        if self.debug_bound_hdlr is not None:
            self.model.catchVarEvent(self.model.getTransformedVar(self.vars["relu_1X30"]),
                                 SCIP_EVENTTYPE.UBTIGHTENED, self.debug_bound_hdlr)


        


