from copy import deepcopy


class LPError(Exception):
    """LP was not solved to stat 1 or 2"""


class LPInfeasible(Exception):
    """LP infeasible"""


class BoundComp():
    """The class BoundComp provides various methods for bound computations w.r.t. neural network verification."""

    def __init__(self, mip):
        self.mip = mip
        self.number_of_inputs = len(self.mip.input_nodes)
        self.lower_bounds = {in_node: [0 if j != i else 1 for j in range(self.number_of_inputs + 1)]
                             for i, in_node in enumerate(self.mip.input_nodes)}
        self.upper_bounds = deepcopy(self.lower_bounds)
        self.last_neuron = None


    def update_symbolic_bounds(self, layer_nodes, use_approximation=True, compare_with_global=True):
        """Update the symbolic equations for the neuron bounds which are maintained in the
        ndarrays self.lower_bounds and self.upper_bounds.

        Args:
            layer_nodes: list of str, the nodes of the layer which shall be updated.
            input_ranges: list of tuples of float, len(input_ranges) == self.number_of_inputs
                            for each input neuron (lb, ub)
            use_approximation: bool, should the Neurify approximation be used or not (old ReluVal method)
            compare_with_global: bool, should symbolic values be compared with other known bounds for the
                                returned (non-symbolic) bounds, must be disabled for calling from model_boundd,
                                because variables to compare do not yet exist

        Return:
              temp_values: dict, str: tuple (lb, ub). For relus, these are the relu_in bounds.

        """

        # TODO include variables fixed by obbt as fixed
        temp_values = {}
        input_ranges = [(self.mip.vars[v].getLbLocal(), self.mip.vars[v].getUbLocal()) for v in self.mip.input_nodes]

        for node_name in layer_nodes:
            suffix = ""
            if node_name in self.mip.relu_nodes:
                relu_in_name, relu_out_name, bias, variable_names, coeffs = self.mip.relu_nodes[node_name]
                suffix = "_in"
            elif node_name in self.mip.linear_nodes:
                bias, variable_names, coeffs = self.mip.linear_nodes[node_name]
            elif node_name in self.mip.max_pool_nodes:
                raise TypeError("no maxpool implemented in domain branching update symbolic bounds")
            else:
                return None

            self.lower_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
            self.upper_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]

            for c, v in zip(coeffs, variable_names):
                for i in range(self.number_of_inputs + 1):
                    if c >= 0:
                        self.lower_bounds[node_name][i] += c * self.lower_bounds[v][i]
                        self.upper_bounds[node_name][i] += c * self.upper_bounds[v][i]
                    else:
                        self.lower_bounds[node_name][i] += c * self.upper_bounds[v][i]
                        self.upper_bounds[node_name][i] += c * self.lower_bounds[v][i]


            self.lower_bounds[node_name][-1] += bias
            self.upper_bounds[node_name][-1] += bias

            # we save only the relu_out values in self.lower_bounds / self.upper_bounds
            # temp_lower and temp_upper are the bounds of the current relu_in
            temp_lower = 0
            temp_upper = 0
            lb_upper_input = 0
            ub_lower_input = 0
            for i in range(self.number_of_inputs):
                if self.lower_bounds[node_name][i] >= 0:
                    temp_lower += self.lower_bounds[node_name][i] * input_ranges[i][0]
                    lb_upper_input += self.lower_bounds[node_name][i] * input_ranges[i][1]
                else:
                    temp_lower += self.lower_bounds[node_name][i] * input_ranges[i][1]
                    lb_upper_input += self.lower_bounds[node_name][i] * input_ranges[i][0]
                if self.upper_bounds[node_name][i] >= 0:
                    temp_upper += self.upper_bounds[node_name][i] * input_ranges[i][1]
                    ub_lower_input += self.upper_bounds[node_name][i] * input_ranges[i][0]
                else:
                    temp_upper += self.upper_bounds[node_name][i] * input_ranges[i][0]
                    ub_lower_input += self.upper_bounds[node_name][i] * input_ranges[i][1]

            # we have to add the whole bias till now, not just the current one
            temp_lower += self.lower_bounds[node_name][-1]
            temp_upper += self.upper_bounds[node_name][-1]
            lb_upper_input += self.lower_bounds[node_name][-1]
            ub_lower_input += self.upper_bounds[node_name][-1]

            #self.old_values[node_name] = temp_lower, temp_upper
            if compare_with_global:
                temp_lower = max(temp_lower, self.mip.vars[node_name + suffix].getLbLocal())
                temp_upper = min(temp_upper, self.mip.vars[node_name + suffix].getUbLocal())
            temp_values[node_name] = temp_lower, temp_upper

            if not use_approximation and node_name in self.mip.relu_nodes:
                if temp_lower < 0:
                    self.lower_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                    # at this point linearity is broken if temp_upper > 0, therefore we replace the
                    # symbolic bound by a fixed bound which is saved as bias
                    self.upper_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                    self.upper_bounds[node_name][-1] = temp_upper
                # this condition eliminates everything and must therefore be the last one
                if temp_upper < 0:
                    self.lower_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                    self.upper_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]

            elif use_approximation and node_name in self.mip.relu_nodes:
                if temp_upper < 0:
                    self.lower_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                    self.upper_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                elif temp_lower > 0:
                    pass
                else:
                    if ub_lower_input < 0:
                        for i in range(self.number_of_inputs + 1):
                            self.upper_bounds[node_name][i] *= temp_upper / (temp_upper - ub_lower_input)
                        self.upper_bounds[node_name][-1] -= temp_upper * ub_lower_input / (temp_upper - ub_lower_input)
                    if lb_upper_input < 0:
                        self.lower_bounds[node_name] = [0 for _ in range(self.number_of_inputs + 1)]
                    else:
                        for i in range(self.number_of_inputs + 1):
                            self.lower_bounds[node_name][i] *= lb_upper_input / (lb_upper_input - temp_lower)

        return temp_values



    def apply_obbt_to_node(self, var, minimize, cutoffbound):
        """Solve the linear max / min LP for the given variable. IMPORTANT: Changes the objective function in
        diving mode, this change must be undone manually afterwards (if desired).

        Args:
            model: SCIP model
            var: scip variable
            minimize: bool, True for minimize, maximize else
            cutoffbound: float, cutoff bound for lp (should be know upper / lower bound of variable)
                            this value is also returned if the LP solving state is not optimal

        Returns:
            float, new bound for the variable

        Raises:
            LPInfeasible:  The current LP is infeasible.
            LPError:       The LP is not solved to optimality, but also not primally infeasible.
        """

        self.mip.model.hideOutput(quiet=False)
        self.mip.model.chgCutoffboundDive(cutoffbound)
        self.mip.model.chgVarObjDive(var, 1 if minimize else -1)
        self.mip.model.constructLP()
        self.mip.model.solveDiveLP()
        stat = self.mip.model.getLPSolstat()
        new_bound = self.mip.model.getLPObjVal() if minimize else -self.mip.model.getLPObjVal()
        self.last_neuron = var

        if stat == 1:
            return new_bound, True
        elif stat == 2:
            raise LPInfeasible
        else:
            raise LPError

    def apply_obbt_two_vars(self, v1, v2, lpsol1, lpsol2):
        """This method may only be called when SCIP is in diving mode. Maximizes v1 + v2
        and adds this as a constraint for the corresponding relu_out vars.

        Args:
            v1: str variable name relu_out
            v2: str variable relu_out
            lpsol1: dict {str: float}, mapping variable name to lp solution value, lp solution corresponds
                    to maximizing v1
            lpsol2: dict {str: float}, mapping variable name to lp solution value, lp solution corresponds
                    to maximizing v2

        """

        x1 = max(0, lpsol1[v1])
        x2 = max(0, lpsol1[v2])
        y1 = max(0, lpsol2[v1])
        y2 = max(0, lpsol2[v2])
        obj1 = y2 - x2
        obj2 = x1 - y1
        neg_val = obj1 * x1 + obj2 * x2

        if obj1 < 0 or obj2 < 0:
            return


        self.mip.model.chgVarObjDive(self.mip.vars[v1 + "_in"], -obj1)
        self.mip.model.chgVarObjDive(self.mip.vars[v2 + "_in"], -obj2)
        self.mip.model.constructLP()
        self.mip.model.solveDiveLP()
        stat = self.mip.model.getLPSolstat()
        if stat != 1:
            if stat == 2:
                raise LPInfeasible
            else:
                raise LPError
        bound_val = -self.mip.model.getLPObjVal()
        if neg_val > bound_val:
            bound_val = neg_val

        if bound_val >= self.mip.eps:
            self.mip.model.addCons(obj1 * self.mip.vars[v1] + obj2 * self.mip.vars[v2] <= bound_val, local=True)
            r = self.mip.model.createEmptyRowUnspec(name=v1 + "_" + v2 + "_obbt_2", lhs=None, rhs=bound_val)
            self.mip.model.addVarToRow(r, self.mip.vars[v1], obj1)
            self.mip.model.addVarToRow(r, self.mip.vars[v2], obj2)
            self.mip.model.addRowDive(r)


        self.mip.model.chgVarObjDive(self.mip.vars[v1 + "_in"], 0)
        self.mip.model.chgVarObjDive(self.mip.vars[v2 + "_in"], 0)


    def compute_linear_bounds(self, variables, coefficients):
        """Copmute the upper and lower bound before the ReLU application to the given variables with the
        corresponding coefficients. Uses the bounds in current dive.

        Args:
            variables: list of str, containing the names of the variables
            coefficients: list of float coefficients
            local: bool, if True local bounds are used, otherwise global bounds

        Returns:
            tuple: (lower bound, upper bound)
        """

        assert len(variables) == len(coefficients)

        lb, ub = 0, 0
        for v_name, c in zip(variables, coefficients):
            v_orig = self.mip.vars[v_name]
            current_lb = self.mip.model.getVarLbDive(v_orig)
            current_ub = self.mip.model.getVarUbDive(v_orig)

            if c > 0:
                ub += current_ub * c
                lb += current_lb * c
            elif c < 0:
                ub += current_lb * c
                lb += current_ub * c

        return lb, ub

