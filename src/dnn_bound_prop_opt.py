from pyscipopt import Prop, SCIP_RESULT, Model, quicksum
import logging
from compute_bounds import BoundComp, LPError, LPInfeasible


class DNNBoundProp(Prop):
    """Contains the variuos propagation methods which are essential for solving neural network verification
    problems."""

    def __init__(self, mip, optimize_nodes, obbt_2, use_symbolic, bound_for_opt, maxdepth, use_genvbounds, obbt_k=2,
                 obbt_l=5, obbt_sort=True):
        self.mip = mip
        self.optimize_nodes = optimize_nodes
        self.obbt_2 = obbt_2
        self.obbt_k = obbt_k
        self.obbt_l = obbt_l
        self.obbt_sort = obbt_sort
        self.last_node_number = -1
        self.bound_comp = BoundComp(self.mip)
        self.vars_approx_lb = {}
        self.log = logging.getLogger('main_log')
        self.final_lower_bounds = {}
        self.final_upper_bounds = {}
        self.use_symbolic = use_symbolic
        self.maxdepth = maxdepth
        self.use_genvbounds = use_genvbounds
        if bound_for_opt < 0:
            self.bound_for_opt = float("inf")
        else:
            self.bound_for_opt = bound_for_opt


    def propexec(self, proptiming):

        if self.mip.model.inProbing() or self.mip.model.inRepropagation():
            return {"result": SCIP_RESULT.DIDNOTRUN}

        current_scip_node = self.mip.model.getCurrentNode()

        if current_scip_node.getNumber() == self.last_node_number:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        else:
            self.last_node_number = current_scip_node.getNumber()

        if current_scip_node.getDepth() > self.maxdepth:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if self.mip.model.isGT(current_scip_node.getLowerbound() - self.mip.eps, 0):
            return {"result": SCIP_RESULT.CUTOFF}

        # skip propagator if counterexample is already found
        if current_scip_node.getNumber() == 1 and self.mip.model.isLT(self.mip.model.getPrimalbound() + self.mip.eps, 0) :
            return {"result": SCIP_RESULT.CUTOFF}


        print("propexec at node", current_scip_node.getNumber(), "at depth", current_scip_node.getDepth())
        self.log.info("propexec at node %i at depth %i, with lower bound %f current global dual bound %f",
                      current_scip_node.getNumber(), current_scip_node.getDepth(),
                      current_scip_node.getLowerbound(), self.mip.model.getDualbound())

        self.mip.model.constructLP()
        self.mip.model.flushLP()
        self.mip.model.startDive()
        if self.use_genvbounds and current_scip_node.getNumber() == 1:
            # this must be called before cons are relaxed
            cutoffrow = self.mip.model.addSingleObjCutoffDive(self.mip.objective_variable, 0.1)

        self.mip.model.relaxAllConssDive()
        self.mip.model.fixAllVariablesToZeroDive()
        self.mip.model.chgVarObjDive(self.mip.objective_variable, 0.0)

        num_fixed = 0
        self.vars = {}
        last_layer_had_relus = False
        bounds = {}
        fixed_pos = set()
        fixed_neg = set()

        for node_name, var in self.mip.input_nodes.items():
            bounds[node_name] = var.getLbGlobal(), var.getUbGlobal()
            self.mip.model.enableVarAndConssDive(var)

        try:

            for layer_index, layer_nodes in enumerate(self.mip.layers):
                if self.use_symbolic:
                    temp_values = self.bound_comp.update_symbolic_bounds(layer_nodes)
                current_layer_has_relus = False

                lp_sols = {}
                layer_bounds = []    # contains only nodes with lb < 0 < ub


                for node_name in layer_nodes:

                    if node_name in self.mip.relu_nodes:

                        current_layer_has_relus = True
                        relu_in_name, relu_out_name, bias, variable_names, coeffs = self.mip.relu_nodes[node_name]
                        lb, ub = self.bound_comp.compute_linear_bounds(variable_names, coeffs)
                        lb += bias - self.mip.eps
                        ub += bias + self.mip.eps

                        relu_out_global = self.mip.vars[relu_out_name]
                        relu_in_global = self.mip.vars[relu_in_name]
                        self.mip.model.enableVarAndConssDive(relu_in_global)
                        self.mip.model.enableVarAndConssDive(relu_out_global)
                        try:
                            self.mip.model.enableVarAndConssDive(self.mip.binary_variables["bin_" + node_name])
                        except KeyError:
                            pass


                        if self.optimize_nodes and last_layer_had_relus and self.mip.model.isFeasGE(ub, 0) \
                                and abs(ub) + abs(lb) < self.bound_for_opt:

                            new_lb, opt = self.bound_comp.apply_obbt_to_node(relu_in_global, True, lb)
                            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                                self.mip.model.createGenVBound(relu_in_global, "genvbounds", cutoffrow, True)
                            self.mip.model.chgVarObjDive(relu_in_global, 0.0)
                            new_ub, opt = self.bound_comp.apply_obbt_to_node(relu_in_global, False, ub)
                            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                                self.mip.model.createGenVBound(relu_in_global, "genvbounds", cutoffrow, False)
                            self.mip.model.chgVarObjDive(relu_in_global, 0.0)
                            # This must be exactly after apply_obbt_to_node with maximizing to get the right lp solution
                            lp_sols[node_name] = {var_name: self.mip.vars[var_name + "_in"].getLPSol() for var_name in layer_nodes}


                        elif self.use_symbolic:
                            new_lb = max(lb, relu_in_global.getLbLocal(), temp_values[node_name][0])
                            new_ub = min(ub, relu_in_global.getUbLocal(), temp_values[node_name][1])

                        else:
                            new_lb = lb
                            new_ub = ub

                        try:
                            assert self.mip.model.isFeasGE(new_lb, lb), str(new_lb) + " < " + str(lb)
                            assert self.mip.model.isFeasLE(new_ub, ub), str(new_ub) + " > " + str(ub)
                        except AssertionError as e:
                            self.log.exception(e)

                        bounds[node_name] = new_lb, new_ub
                        #print(node_name, new_lb, relu_in_global.getLbLocal(), new_ub, relu_in_global.getUbLocal())

                        new_lb -= self.mip.eps
                        new_ub += self.mip.eps

                        new_ub_relu_out = new_ub if new_ub > 0 else 0

                        # chgVarLbTighten /chgVarUbTighten also tightens the bounds in dive mode, though addCons
                        # is NOT applied in the dive mode

                        self.mip.model.chgVarLbTighten(relu_in_global, new_lb)
                        self.mip.model.chgVarUbTighten(relu_in_global, new_ub)
                        self.mip.model.chgVarUbTighten(relu_out_global, new_ub_relu_out)

                        # here we add the corresponding constraints, depending on the bound values
                        if self.mip.model.isLT(new_ub, 0):
                            fixed_neg.add(node_name)
                            num_fixed += 1
                            try:
                                self.mip.model.chgVarUbTighten(self.mip.binary_variables["bin_" + node_name], 0)
                                self.mip.model.delConsLocal(self.mip.relu_cons[node_name][current_scip_node.getNumber()][0])
                                self.mip.model.delConsLocal(self.mip.relu_cons[node_name][current_scip_node.getNumber()][1])
                            except KeyError:  # if the variable was fixed in the beginning already, binary
                                pass  # variable is not created


                        elif self.mip.model.isGT(new_lb, 0):
                            fixed_pos.add(node_name)
                            num_fixed += 1
                            self.mip.model.addCons(relu_out_global == relu_in_global,
                                                   name=node_name + "_fix_pos_node" + str(current_scip_node.getNumber()),
                                                   local=True)
                            try:
                                self.mip.model.chgVarLbTighten(self.mip.binary_variables["bin_" + node_name], 1)
                                self.mip.model.delConsLocal(self.mip.relu_cons[node_name][current_scip_node.getNumber()][0])
                                self.mip.model.delConsLocal(self.mip.relu_cons[node_name][current_scip_node.getNumber()][1])
                            except KeyError:  # if the variable was fixed in the beginning already, binary
                                pass  # variable is not created

                            self.mip.model.chgVarLbTighten(relu_out_global, new_lb)

                        else:
                            try:
                                d = self.mip.binary_variables["bin_" + node_name]
                                c1 = self.mip.model.addCons(relu_out_global <= relu_in_global - (1 - d) * new_lb,
                                                            name=node_name + "_bin_lb_node" + str(
                                                                current_scip_node.getNumber()),
                                                            local=True)
                                c2 = self.mip.model.addCons(relu_out_global <= d * new_ub,
                                                            name=node_name + "_bin_ub_node" + str(
                                                                current_scip_node.getNumber()),
                                                            local=True)
                                r1 = self.mip.model.createEmptyRowUnspec(name=node_name + "_bin_lb_node_dive", lhs=None, rhs=-new_lb)
                                self.mip.model.addVarToRow(r1, relu_out_global, 1)
                                self.mip.model.addVarToRow(r1, relu_in_global, -1)
                                self.mip.model.addVarToRow(r1, d, -new_lb)

                                r2 = self.mip.model.createEmptyRowUnspec(name=node_name + "_bin_ub_node_dive", lhs=None, rhs=0)
                                self.mip.model.addVarToRow(r2, relu_out_global, 1)
                                self.mip.model.addVarToRow(r2, d, -new_ub)

                                self.mip.model.addRowDive(r1)
                                self.mip.model.addRowDive(r2)


                                if current_scip_node.getNumber() >= 2:
                                    try:
                                        scip_node = current_scip_node.getParent()
                                        while scip_node.getNumber() not in self.mip.relu_cons[node_name] and scip_node.getNumber() > 1:
                                            scip_node = scip_node.getParent()
                                        self.mip.model.delConsLocal(self.mip.relu_cons[node_name]
                                                                    [scip_node.getNumber()][0])
                                        self.mip.model.delConsLocal(self.mip.relu_cons[node_name]
                                                                    [scip_node.getNumber()][1])
                                    except KeyError:
                                        pass
                                self.mip.relu_cons[node_name][current_scip_node.getNumber()] = (c1, c2)
                            except KeyError:
                                print("no binary var there")
                                pass

                            layer_bounds.append((node_name, new_lb, new_ub))


                        if current_scip_node.getDepth() == 0:
                            self.final_lower_bounds[node_name] = min(new_lb, self.final_lower_bounds.get(node_name, float("inf")))
                            self.final_upper_bounds[node_name] = max(new_ub, self.final_upper_bounds.get(node_name, float("-inf")))


                    elif node_name in self.mip.linear_nodes:

                        bias, variable_names, coeffs = self.mip.linear_nodes[node_name]
                        lb, ub = self.bound_comp.compute_linear_bounds(variable_names, coeffs)
                        new_lb = lb + bias - self.mip.eps
                        new_ub = ub + bias + self.mip.eps
                        linear = self.mip.vars[node_name]
                        self.mip.model.enableVarAndConssDive(linear)

                        if self.optimize_nodes and last_layer_had_relus and abs(ub) + abs(lb) < self.bound_for_opt:
                            new_lb, opt = self.bound_comp.apply_obbt_to_node(linear, True, lb)
                            new_lb -= self.mip.eps
                            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                                self.mip.model.createGenVBound(linear, "genvbounds", cutoffrow, True)
                            self.mip.model.chgVarObjDive(linear, 0.0)
                            new_ub, opt = self.bound_comp.apply_obbt_to_node(linear, False, ub)
                            new_ub += self.mip.eps
                            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                                self.mip.model.createGenVBound(linear, "genvbounds", cutoffrow, False)
                            self.mip.model.chgVarObjDive(linear, 0.0)

                        self.mip.model.chgVarLbTighten(linear, new_lb)
                        self.mip.model.chgVarUbTighten(linear, new_ub)
                        #print(node_name, lb+bias, ub+bias, new_lb, new_ub)
                        bounds[node_name] = new_lb, new_ub


                        if current_scip_node.getDepth() == 0:
                            self.final_lower_bounds[node_name] = min(new_lb,
                                                                     self.final_lower_bounds.get(node_name, float("inf")))
                            self.final_upper_bounds[node_name] = max(new_ub,
                                                                     self.final_upper_bounds.get(node_name, float("-inf")))


                    elif node_name in self.mip.max_pool_nodes:

                        raise TypeError("no maxpool implemented in domain branching")

                    else:
                        current_layer_has_relus = last_layer_had_relus    # transport the information to the next "real" layer
                        break   # layer can be skipped (relu_in), go to next layer.


                last_layer_had_relus = current_layer_has_relus

                if self.obbt_2 and layer_bounds and lp_sols:
                    if self.obbt_sort:
                        layer_bounds = sorted(layer_bounds, key=lambda x: x[2]*x[1] / (x[2] - x[1]))
                    start_index = 0

                    for round in range(self.obbt_k):
                        layer_bounds = layer_bounds[start_index:]
                        selected = None
                        for el in layer_bounds:
                            start_index += 1
                            if el[0] in lp_sols:
                                selected = el[0]
                                break

                        if selected is None:
                            break
                        ranking = []
                        # it holds l_sel < 0 < u_sel due to filling of layer_bounds
                        l_sel = self.mip.model.getVarLbDive(self.mip.vars[selected + "_in"])
                        u_sel = self.mip.model.getVarUbDive(self.mip.vars[selected + "_in"])
                        for n, val_n in lp_sols[selected].items():
                            if n == selected:
                                continue

                            l_n = self.mip.model.getVarLbDive(self.mip.vars[n + "_in"])
                            u_n = self.mip.model.getVarUbDive(self.mip.vars[n + "_in"])
                            
                            rank_sel = 0
                            try:
                                val_sel = lp_sols[n][selected]
                                if val_sel >= 0:
                                    rank_sel = (val_sel*u_sel - u_sel*l_sel) / (u_sel - l_sel) - val_sel
                                else:
                                    rank_sel = (val_sel*u_sel - u_sel*l_sel) / (u_sel - l_sel)
                            except KeyError:
                                pass
                                
                            if l_n < -self.mip.eps and self.mip.eps < u_n:
                                if val_n >= 0:
                                    ranking.append((rank_sel + (val_n*u_n - u_n*l_n) / (u_n - l_n) - val_n, n))
                                else:
                                    ranking.append((rank_sel + (val_n*u_n - u_n*l_n) / (u_n - l_n), n))

                        if self.obbt_sort:
                            ranking = sorted(ranking, key=lambda x: x[0], reverse=True)

                        for i in range(1, min(self.obbt_l, len(ranking))):
                            try:
                                self.bound_comp.apply_obbt_two_vars(selected, ranking[i][1],
                                                                lp_sols[selected], lp_sols[ranking[i][1]])
                            except KeyError:
                                pass    # if variable was not optimized (i.e. symbolic bounds used), no lp solution exists



            self.log.debug("fixed %i variables", num_fixed)

            # update of branch candidate list for ReLU branching
            try:
                counter = 0
                for i in range(len(self.mip.nodes_by_branch_prio[current_scip_node.getNumber()])):
                    if self.mip.nodes_by_branch_prio[current_scip_node.getNumber()][i] in fixed_neg.union(fixed_pos):
                        self.mip.nodes_by_branch_prio[current_scip_node.getNumber()][i] = None
                        counter += 1
                self.mip.nodes_by_branch_prio[current_scip_node.getNumber()] = [u for u in
                            self.mip.nodes_by_branch_prio[current_scip_node.getNumber()] if u is not None]
            except KeyError:
                pass


            upper_bounds = {}
            lower_bounds = {}

            for v in self.mip.output_variables_binary:
                self.mip.model.enableVarAndConssDive(v)

            for out_var_name, (variable_names, coeffs, operator, lhs) in self.mip.output_variables.items():
                lb, ub = self.bound_comp.compute_linear_bounds(variable_names, coeffs)
                lb -= self.mip.eps
                ub += self.mip.eps

                out_var_global = self.mip.vars[out_var_name]
                self.mip.model.enableVarAndConssDive(out_var_global)

                # the output variables are simply created by affine transformations, therefore the bounds
                # computed by self.compute_linear_bounds should be tight, hence we don't solve an LP

                if operator:
                    self.mip.model.chgVarLbTighten(out_var_global, lb - lhs)
                    self.mip.model.chgVarUbTighten(out_var_global, ub - lhs)
                    lower_bounds[out_var_name] = (lb - lhs)
                    upper_bounds[out_var_name] = (ub - lhs)

                else:
                    self.mip.model.chgVarLbTighten(out_var_global, lhs - ub)
                    self.mip.model.chgVarUbTighten(out_var_global, lhs - lb)
                    lower_bounds[out_var_name] = (lhs - ub)
                    upper_bounds[out_var_name] = (lhs - lb)


            self.mip.model.enableVarAndConssDive(self.mip.objective_variable)
            M = max(upper_bounds.values()) + self.mip.eps
            for out_var_name in self.mip.output_variables:
                _x = self.mip.vars[out_var_name]
                _d = self.mip.binary_variables["bin_max_pool_opt_" + out_var_name]
                c = self.mip.model.addCons(self.mip.objective_variable <=
                                           _x + (M - lower_bounds[out_var_name] + self.mip.eps) * (1 - _d),
                                           name="max_pool_opt_bin_ub_" + out_var_name + "_node_" + str(
                                               current_scip_node.getNumber()),
                                           local=True)


            new_lb, opt = self.bound_comp.apply_obbt_to_node(self.mip.objective_variable, True, max(lower_bounds.values()))
            new_lb -= self.mip.eps
            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                self.mip.model.createGenVBound(self.mip.objective_variable, "genvbounds", cutoffrow, True)
            self.mip.model.chgVarObjDive(self.mip.objective_variable, 0.0)
            new_ub, opt = self.bound_comp.apply_obbt_to_node(self.mip.objective_variable, False, max(upper_bounds.values()))
            new_ub += self.mip.eps
            if current_scip_node.getNumber() == 1 and self.use_genvbounds:
                self.mip.model.createGenVBound(self.mip.objective_variable, "genvbounds", cutoffrow, False)
            self.mip.model.chgVarObjDive(self.mip.objective_variable, 0.0)

            self.mip.model.chgVarLbTighten(self.mip.objective_variable, new_lb)
            self.mip.model.chgVarUbTighten(self.mip.objective_variable, new_ub)

            self.log.debug("optimized t %f %f", new_lb, new_ub)

            if self.mip.verify_or_constraints:
                new_lb, new_ub = -new_ub, -new_lb   # switch bounds for NodeLowerbound, not relevant for Maxpool cons

            if new_lb >= current_scip_node.getLowerbound():
                # in case of or constraints the old t lb is actually the node upper bound
                self.log.debug("node bound update, old node lb %f old t lb (maybe node ub) %f, new lb is %f",
                               current_scip_node.getLowerbound(), self.mip.objective_variable.getLbLocal(), new_lb)
                if self.mip.model.isGT(new_lb - self.mip.eps, 0.0):
                    self.mip.model.endDive()
                    self.log.debug("Node cut off.")
                    return {"result": SCIP_RESULT.CUTOFF}
                self.mip.model.updateNodeLowerbound(current_scip_node, new_lb)
                self.mip.model.updateNodeDualbound(current_scip_node, new_lb)
            else:
                self.log.debug("node bound NOT updated, old node lb %f, new lb would be %f",
                        current_scip_node.getLowerbound(), new_lb)

            self.mip.model.freeRowsSidesArrays()
            self.mip.model.endDive()

            return {"result": SCIP_RESULT.REDUCEDDOM}

        except LPError:
            # this should ideally never occur
            self.mip.model.endDive()
            self.log.warning("LPError occured. Return DIDNOTRUN")
            return {"result": SCIP_RESULT.DIDNOTRUN}

        except LPInfeasible:
            # this case is normal behaviour, may result from primal solutions found by heuristic
            self.mip.model.endDive()
            self.log.info("LPInfeasible occured. Return CUTOFF")
            return {"result": SCIP_RESULT.DIDNOTRUN}
