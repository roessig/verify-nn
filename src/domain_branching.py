from pyscipopt import Branchrule, SCIP_RESULT, Model, quicksum
import torch
import logging
from copy import deepcopy


class DomainBranching(Branchrule):
    """Class to perform domain branching as introduced by Bunel et al."""

    def __init__(self, mip, opt_mode, split_mode):

        self.mip = mip
        self.opt_mode = opt_mode
        self.branch_nodes = None
        self.split_mode = split_mode
        self.log = logging.getLogger('main_log')
        self.number_of_inputs = len(self.mip.input_nodes)
        self.lower_bounds = {in_node: [0 if j != i else 1 for j in range(self.number_of_inputs + 1)]
                             for i, in_node in enumerate(self.mip.input_nodes)}
        self.upper_bounds = deepcopy(self.lower_bounds)
        self.fixed_neurons = {}
        self.branching_variables = {}

    def quicksum_from_var_names(self, variable_names, coeffs):
        """Returns a quicksum given only the variables' names and the coeffs."""
        return quicksum(c * self.vars[v] for v, c in zip(*(variable_names, coeffs)))


    def compute_linear_bounds(self, variables, coefficients):
        """Copmute the upper and lower bound before the ReLU application to the given variables with the
        corresponding coefficients.

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
            current_lb = v_orig.getLbLocal()
            current_ub = v_orig.getUbLocal()

            if c > 0:
                ub += current_ub * c
                lb += current_lb * c
            elif c < 0:
                ub += current_lb * c
                lb += current_ub * c

        return lb, ub


    def branchexeclp(self, allowaddcons):
        """Execute branching based on LP solution."""

        assert allowaddcons, "allowaddcons is False in DomainBranching, we have to terminate"
        current_scip_node = self.mip.model.getCurrentNode()
        self.log.info("DomainBranching at node %i, depth %i, with lower bound %f",
                      current_scip_node.getNumber(), current_scip_node.getDepth(),
                     current_scip_node.getLowerbound())


        # check whether the current domain is already proven to be insatisfiable
        if self.opt_mode and (self.mip.objective_variable.getLbLocal() >= self.mip.eps or
                current_scip_node.getLowerbound() >= self.mip.eps):
            self.log.info("skip branching due to positive lb of t or node")
            return {"result": SCIP_RESULT.DIDNOTRUN}

        elif self.opt_mode and self.mip.objective_variable.getUbLocal() <= -self.mip.eps:  # counterexample found
            self.log.info("skip branching due to negative ub")
            self.log.debug("ub local %f, global %f", self.mip.objective_variable.getUbLocal(),
                           self.mip.objective_variable.getUbGlobal())
            self.log.debug("lb local %f, global %f", self.mip.objective_variable.getLbLocal(),
                           self.mip.objective_variable.getLbGlobal())

            return {"result": SCIP_RESULT.DIDNOTRUN}

        # we do branching to improve the bounds
        else:
            split_ranges = {}
            down_ranges = {}
            up_ranges = {}
            for neuron_name, neuron in self.mip.input_nodes.items():
                lb = neuron.getLbLocal()
                ub = neuron.getUbLocal()

                split_ranges[neuron_name] = ub - lb
                down_ranges[neuron_name] = lb, ub
                up_ranges[neuron_name] = lb, ub
            #self.log.debug("Input neuron bounds before branching " + str(up_ranges))

            if self.branch_nodes is None:
                self.branch_nodes = list(self.mip.input_nodes.keys())


            if self.split_mode == "standard":
                split_neuron_name = self.branch_nodes[current_scip_node.getDepth() % self.number_of_inputs]

            elif self.split_mode == "gradient":
                # compute the input neuron on which splitting will have the highest impact
                # according to the gradient of the NN at this input neuron
                sum_of_gradients = torch.zeros(len(self.mip.input_nodes))
                value_dict = {in_name: (in_var.getLbLocal(), (in_var.getUbLocal() - in_var.getLbLocal()) / 2
                                        + in_var.getLbLocal(), in_var.getUbLocal())
                              for in_name, in_var in self.mip.input_nodes.items()}

                for i in range(3):
                    input_tensor = torch.tensor([val[i] for val in value_dict.values()], requires_grad=True)
                    torch_res = self.mip.pytorch_model(input_tensor)
                    torch_res.backward()
                    sum_of_gradients += input_tensor.grad

                res_list = [(abs(float(x)) * (v.getUbLocal() - v.getLbLocal()), v_name)
                            for x, (v_name, v) in zip(sum_of_gradients, self.mip.input_nodes.items())
                            if split_ranges[v_name] >= self.mip.eps]

                # may happen if all split ranges are very small / 0
                if not res_list:
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                max_res = max(res_list)
                if max_res[0] == min(res_list)[0]:  # if value of min gradient = max gradient, use "standard" rule
                    inc = 0
                    while split_ranges[self.branch_nodes[current_scip_node.getDepth() % self.number_of_inputs + inc]] < self.mip.eps:
                        inc += 1
                        if inc >= self.number_of_inputs:
                            return {"result": SCIP_RESULT.DIDNOTRUN}

                    split_neuron_name = self.branch_nodes[current_scip_node.getDepth() % self.number_of_inputs + inc]
                else:
                    split_neuron_name = max_res[1]
                if current_scip_node.getDepth() >= 2:
                    parent = current_scip_node.getParent()
                    if split_neuron_name == self.branching_variables[parent.getNumber()] and \
                                split_neuron_name == self.branching_variables[parent.getParent().getNumber()]:
                        split_neuron_name = sorted(res_list)[-2][1]

                self.log.debug("averaged gradient: " + str(sum_of_gradients))
                # self.log.debug("res_list gradient mode: " + str(res_list))

            else:
                raise AttributeError("Unknown split mode for domain branching.")

            self.branching_variables[current_scip_node.getNumber()] = split_neuron_name
            split_range = split_ranges[split_neuron_name]

            counter = 0
            while split_range < self.mip.eps:
                counter += 1
                split_neuron_name = self.branch_nodes[(current_scip_node.getDepth() + counter) % self.number_of_inputs]
                split_range = split_ranges[split_neuron_name]

                if counter >= self.number_of_inputs:
                    return {"result": SCIP_RESULT.DIDNOTRUN}

            split_neuron = self.mip.input_nodes[split_neuron_name]
            split_point = split_range / 2 + split_neuron.getLbLocal()

            down_ranges[split_neuron_name] = (split_neuron.getLbLocal(), split_point)
            up_ranges[split_neuron_name] = (split_point, split_neuron.getUbLocal())

            down, eq, up = self.mip.model.branchVarVal(split_neuron, split_point)
            self.log.info("Ready to branch at depth %i with node number %i. Branch variable"
                          " %s at split point %f with range %f",
                          current_scip_node.getDepth(), current_scip_node.getNumber(),
                          split_neuron_name, split_point, split_range)

        return {"result": SCIP_RESULT.BRANCHED}


    def branchexecext(self, allowaddcons):
        print("Relu branch ext", allowaddcons)
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def branchexecps(self, allowaddcons):
        print("Relu branch not all fixed", allowaddcons)
        return {"result": SCIP_RESULT.DIDNOTRUN}