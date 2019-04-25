from pyscipopt import Heur, SCIP_RESULT
import random
import torch
import logging
from copy import deepcopy
from itertools import chain


class SamplingHeuristic(Heur):
    """Contains heuristics for finding primal solutions in the neural network verification problem."""

    def __init__(self, mip, max_iter, opt_mode, use_local_bounds=False, bound_for_lp_heur=0.1, max_iter_lp_heur=10,
                 use_lp_sol_gen=False, use_relu_branch_gradient=False):

        self.max_iter = max_iter
        self.mip = mip
        self.opt_mode = opt_mode
        self.use_local_bounds = use_local_bounds
        self.bound_for_lp_heur = bound_for_lp_heur if bound_for_lp_heur > 0 else -1
        self.log = logging.getLogger('main_log')
        self.max_iter_lp_heur = max_iter_lp_heur
        self.use_lp_sol_gen = use_lp_sol_gen
        self.use_relu_branch_gradient = use_relu_branch_gradient


    def _get_input_values_local(self):
        """Simple sampling method (only lower and upper bounds for each input variable) using the local bounds."""

        sample_values = {}
        for name, in_var in self.mip.input_nodes.items():
            lb = in_var.getLbLocal()
            ub = in_var.getUbLocal()
            sample_values[name] = random.random() * (ub - lb) + lb
        return sample_values


    def _get_input_values_global(self):
        """Simple sampling method (only lower and upper bounds for each input variable) using the global bounds."""

        sample_values = {}
        for name, in_var in self.mip.input_nodes.items():
            lb = in_var.getLbGlobal()
            ub = in_var.getUbGlobal()
            sample_values[name] = random.random() * (ub - lb) + lb
        return sample_values


    def _get_input_values_lp(self):
        """Use this method if general linear constraints are imposed on the inputs.
        Sample input values that are feasible for the input bounds by solving an LP."""

        sample_values = {}
        for var in self.mip.input_nodes.values():
            self.mip.model.chgVarObjDive(var, 1 - 2 * random.random())
        self.mip.model.solveDiveLP()
        for v_name, var in self.mip.input_nodes.items():
            sample_values[v_name] = self.mip.model.getSolVal(None, var)

        return sample_values


    def heurexec(self, heurtiming, nodeinfeasible):
        """Execute the heuristic."""

        self.log.debug("sampling heurexec, local: %i", self.use_local_bounds)

        relu_phases = {x: -1 for x in self.mip.relu_nodes}

        has_found_solution = False
        min_output = float("inf")
        min_input = None

        current_scip_node = self.mip.model.getCurrentNode()
        current_node_num = current_scip_node.getNumber()
        self.mip.nodes_by_branch_prio[current_node_num] = [n for n in self.mip.nodes_by_branch_prio[current_scip_node.
            getParent().getNumber()] if n is not None] if current_node_num > 1 else list(relu_phases.keys())
        branch_prios = {n: 0 for n in self.mip.nodes_by_branch_prio[current_node_num] if n is not None}

        if self.max_iter <= 0:
            return {"result": SCIP_RESULT.DIDNOTFIND}

        if self.use_lp_sol_gen:
            self.mip.model.constructLP()
            self.mip.model.flushLP()
            self.mip.model.startDive()
            self.mip.model.relaxAllConssDive()
            self.mip.model.fixAllVariablesToZeroDive()
            if self.opt_mode:
                self.mip.model.chgVarObjDive(self.mip.objective_variable, 0.0)
            for var in self.mip.input_nodes.values():
                self.mip.model.enableVarAndConssDive(var)


        for i in range(self.max_iter):
            if self.use_local_bounds and not self.use_lp_sol_gen:
                sample_values = self._get_input_values_local()
            elif not self.use_local_bounds and not self.use_lp_sol_gen:
                sample_values = self._get_input_values_global()
            else:
                sample_values = self._get_input_values_lp()

            # perform random sampling using the pytorch model
            inputs_sorted = [v for n, v in sorted(sample_values.items(), key=lambda x: x[0])]
            input_tensor = torch.tensor(inputs_sorted, requires_grad=True)
            torch_res = self.mip.pytorch_model(input_tensor)
            current_out = torch_res.item()
            if self.mip.verify_or_constraints:
                current_out = - current_out
            if current_out < min_output:
                min_input = deepcopy(sample_values)
                min_output = current_out

            # determine branching priorities if gradient is used for ReLU branching
            if self.use_relu_branch_gradient and i % 100 == 0:
                torch_res.backward()
                for n in self.mip.nodes_by_branch_prio[current_node_num]:
                    if n is not None:
                        i, j = self.mip.node_position_pytorch[n]
                        for ind, el in enumerate(self.mip.pytorch_model.parameters()):
                            if 2*i == ind:
                                branch_prios[n] += abs(sum(el.grad[j, :]).item())

        if self.use_relu_branch_gradient:
            self.mip.nodes_by_branch_prio[current_node_num] = sorted(self.mip.nodes_by_branch_prio[current_node_num],
                                                        key=lambda x: branch_prios[x], reverse=True)
            print([(x, branch_prios[x]) for x in self.mip.nodes_by_branch_prio[current_node_num]])

        if self.use_lp_sol_gen:
            self.mip.model.endDive()

        # min_output is already inverted, i.e. with correct sign, in case of verify_or_constraints
        if self.mip.model.isGE(min_output, self.mip.model.getPrimalbound()):
            self.log.debug("bound not met in sampling heurexec %f (maybe inverted)", min_output)
            # One could abort the heuristic here to reduce the time spent on the heuristic. On the other hand,
            # in some cases, better primal solutions could be found by continuing with the LP heuristic.
            #return {"result": SCIP_RESULT.DIDNOTFIND}

        list_of_min_inputs = [min_input]
        for min_input in list_of_min_inputs:

            value_dict = min_input
            sol = self.mip.model.createSol()
            value_dict_opt = {}

            for node in self.mip.nodes_sorted:
                new_value = 0

                if node in self.mip.input_nodes:
                    continue  # skip the input nodes

                elif node in self.mip.linear_nodes or node in self.mip.relu_in_nodes:
                    for n in self.mip.graph.predecessors(node):
                        new_value += self.mip.graph.edges[n, node]["weight"] * value_dict[n]

                    new_value += self.mip.graph.node[node]["bias"]

                elif node in self.mip.relu_nodes:
                    pred = list(self.mip.graph.predecessors(node))
                    assert len(pred) == 1    # the corresponding relu_in node should be the only predecessor

                    if node in self.mip.fixed_negative:
                        # if relu_in is > 0, we can abort, otherwise new_value is already set to 0
                        if not self.mip.model.isFeasLE(value_dict[pred[0]], 0):
                            raise ValueError
                        else:
                            relu_phases[node] = 0
                    elif node in self.mip.fixed_positive:
                        if not self.mip.model.isFeasGE(value_dict[pred[0]], 0):
                            raise ValueError
                        else:
                            relu_phases[node] = 1
                            new_value = value_dict[pred[0]]

                    else:  # normal case, node is not fixed yet
                        if value_dict[pred[0]] > 0:  # apply ReLU here
                            new_value = value_dict[pred[0]]
                            relu_phases[node] = 1
                            self.mip.model.setSolVal(sol, self.mip.binary_variables["bin_" + node], 1)

                        else:
                            relu_phases[node] = 0    # new_value is initialized with 0 already
                            self.mip.model.setSolVal(sol, self.mip.binary_variables["bin_" + node], 0)

                else:
                    raise AttributeError("Inexistent neuron.")

                value_dict[node] = new_value

            if self.opt_mode:

                for node in self.mip.output_variables:    # this contains the linear opt nodes
                    new_value = 0

                    for n in self.mip.graph.predecessors(node):
                        new_value += self.mip.graph.edges[n, node]["weight"] * value_dict[n]

                    new_value += self.mip.graph.node[node]["bias"]
                    value_dict_opt[node] = new_value
                    self.mip.model.setSolVal(sol, self.mip.vars[node], new_value)

                # set value of objective variable "t"
                t_value = float("-inf")
                argmax = None
                for n in self.mip.graph.predecessors(self.mip.objective_variable.name):
                    if value_dict_opt[n] > t_value:
                        t_value = value_dict_opt[n]
                        argmax = n

                self.mip.model.setSolVal(sol, self.mip.objective_variable, t_value)
                for n in self.mip.graph.predecessors(self.mip.objective_variable.name):
                    if n == argmax:
                        self.mip.model.setSolVal(sol, self.mip.binary_variables["bin_max_pool_opt_" + n], 1)
                    else:
                        self.mip.model.setSolVal(sol, self.mip.binary_variables["bin_max_pool_opt_" + n], 0)

                output_value = -t_value if self.mip.verify_or_constraints else t_value


                for node, value in value_dict.items():
                    self.mip.model.setSolVal(sol, self.mip.vars[node], value)


                if 0 <= output_value < self.bound_for_lp_heur:  #not self.use_local_bounds and
                    self.mip.model.constructLP()
                    self.mip.model.flushLP()
                    self.mip.model.startDive()

                    # this is necessary! Otherwise a problem occurs, although these lines seem unreasonable.
                    self.mip.model.chgVarObjDive(self.mip.objective_variable, 0.0)
                    self.mip.model.chgVarObjDive(self.mip.objective_variable, -1.0 if self.mip.verify_or_constraints else 1.0)

                    for bin_var in self.mip.binary_variables.values():

                        val = self.mip.model.getSolVal(sol, bin_var)
                        #print(bin_var.name, self.mip.model.getSolVal(None, self.mip.vars[bin_var.name[4:] + "_in"]), val)

                        if val == 1:
                            self.mip.model.chgVarLbDive(bin_var, 1.0)
                            self.mip.model.chgVarUbDive(bin_var, 1.0)
                        elif val == 0:
                            self.mip.model.chgVarUbDive(bin_var, 0.0)
                            self.mip.model.chgVarLbDive(bin_var, 0.0)
                        else:
                            raise ValueError

                    self.mip.model.solveDiveLP()
                    all_switches = set()

                    lps_feasible = True
                    for i in range(self.max_iter_lp_heur):
                        phase_switch_candidates = []
                        if self.mip.model.getLPSolstat() != 1:
                            lps_feasible = False
                            break

                        for node, value in value_dict.items():
                            sol_val = self.mip.model.getSolVal(None, self.mip.vars[node])
                            if node in self.mip.relu_in_nodes and self.mip.model.isFeasZero(sol_val) and \
                                    "bin_" + node[:-3] in self.mip.binary_variables:
                                phase_switch_candidates.append((node, self.mip.model.getSolVal(None, self.mip.binary_variables["bin_" + node[:-3]])))

                        if not phase_switch_candidates:
                            break
                        all_switches.update(set(phase_switch_candidates))
                        node, old_phase = random.choice(phase_switch_candidates)
                        assert self.mip.model.isFeasEQ(old_phase, 1) or self.mip.model.isFeasZero(old_phase)

                        # switch the value of the binary variable and solve again
                        bin_var = self.mip.binary_variables["bin_" + node[:-3]]
                        self.mip.model.chgVarUbDive(bin_var, 1 - old_phase)
                        self.mip.model.chgVarLbDive(bin_var, 1 - old_phase)
                        self.mip.model.setSolVal(sol, bin_var, 1 - old_phase)
                        self.mip.model.solveDiveLP()
                        stat = self.mip.model.getLPSolstat()
                        if stat != 1:
                            print("not feasible")
                            lps_feasible = False
                        else:
                            if self.mip.model.isLT(self.mip.model.getLPObjVal(), -self.mip.eps):
                                break


                    # we use the last LP solution as solution of the heuristic, if it is probably better than
                    # the previous solution generated by the heuristic
                    #if lps_feasible and self.mip.model.isFeasLT(self.mip.model.getLPObjVal(), output_value):
                    if lps_feasible:
                        print("change solution to lp heur sol")
                        for node in chain(value_dict, self.mip.output_variables):
                            self.mip.model.setSolVal(sol, self.mip.vars[node], self.mip.model.getSolVal(None, self.mip.vars[node]))
                        self.mip.model.setSolVal(sol, self.mip.objective_variable, self.mip.model.getSolVal(None, self.mip.objective_variable))
                    else:
                        self.log.warning("Sampling Heurexec lps_feasible = False")
                        self.mip.model.endDive()
                        return {"result": SCIP_RESULT.DIDNOTFIND}

                    self.log.debug("After LP heur: first value %f, last value %f", output_value, self.mip.model.getLPObjVal())
                    output_value = self.mip.model.getLPObjVal()
                    self.mip.model.endDive()

                else:    # not using LP heuristic
                    # set solution values for all non-binary and non-opt variables, which are already set above
                    for node, value in value_dict.items():
                        self.mip.model.setSolVal(sol, self.mip.vars[node], value)
            else:   # not opt_mode
                # set solution values for all non-binary and non-opt variables, which are already set above
                for node, value in value_dict.items():
                    self.mip.model.setSolVal(sol, self.mip.vars[node], value)

            solution_accepted = self.mip.model.trySol(sol, printreason=False, checkintegrality=False)

            # in the first two cases the problem is solved
            if solution_accepted and not self.opt_mode:
                self.mip.model.interruptSolve()
                return {"result": SCIP_RESULT.FOUNDSOL}
            elif solution_accepted and self.opt_mode and output_value < -self.mip.eps:
                self.mip.model.interruptSolve()
                return {"result": SCIP_RESULT.FOUNDSOL}
            # in this case we only get a primal bound for the optimization version
            elif solution_accepted and self.opt_mode and output_value >= 0:
                has_found_solution = True


        # reached max_iter without finding an actual solution, which is a counterexample for the problem
        if has_found_solution:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

