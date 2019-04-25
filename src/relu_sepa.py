from pyscipopt import Conshdlr, SCIP_RESULT, quicksum, Sepa
import logging


class ReluSepa(Sepa):

    def __init__(self, mip):
        self.mip = mip
        self.last_node = None
        self.log = logging.getLogger('main_log')


    def _check(self, sol):
        """Execute the separation method of Anderson et al. and add cuts to the model, if possible."""

        separated = False
        current_scip_node = self.mip.model.getCurrentNode()
        if current_scip_node.getNumber() == self.last_node or current_scip_node.getDepth() < 1:
            self.last_node = current_scip_node.getNumber()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        else:
            self.last_node = current_scip_node.getNumber()
            print("sepa exe", self.last_node)
            self.log.debug("execute separator at node number %i, depth %i", current_scip_node.getNumber(), current_scip_node.getDepth())

        for el in self.mip.relu_nodes.values():
            relu_in_name, relu_out_name, bias, variable_names, coeffs = el

            try:
                z_var = self.mip.binary_variables["bin_" + relu_out_name]
                z_val = self.mip.model.getSolVal(sol, z_var)
                y_val = self.mip.model.getSolVal(sol, self.mip.vars[relu_out_name])
            except KeyError:
                continue      # skip if the ReLU is already fixed

            # definition of variables as in Anderson et al.
            i_set = []
            non_i_set = []
            l_hat = {}
            u_hat = {}
            x_vals = {}
            for c, v in zip(coeffs, variable_names):

                if c > 0:
                    l_hat[v] = self.mip.vars[v].getLbLocal()
                    u_hat[v] = self.mip.vars[v].getUbLocal()
                elif c < 0:
                    l_hat[v] = self.mip.vars[v].getUbLocal()
                    u_hat[v] = self.mip.vars[v].getLbLocal()
                else:
                    continue
                x_val = self.mip.model.getSolVal(sol, self.mip.vars[v])
                x_vals[v] = x_val
                if c * x_val < c * (l_hat[v] * (1 - z_val) + u_hat[v] * z_val):
                    i_set.append((c, v))
                else:
                    non_i_set.append((c, v))

            res = 0
            for c, v in i_set:
                res += c * (x_vals[v] - l_hat[v] * (1 - z_val))
            for c, v in non_i_set:
                res += c * u_hat[v] * z_val

            # check the condition of the separation method
            if self.mip.model.isGT(y_val, bias * z_val + res):

                row_rhs = -sum(c * l_hat[v] for c, v in i_set)
                row = self.mip.model.createEmptyRowSepa(self, name="ideal_relu_" + relu_out_name,
                                                        lhs=-self.mip.model.infinity(), rhs=row_rhs)
                self.mip.model.addVarToRow(row, self.mip.vars[relu_out_name], 1)
                coef_z = - bias + row_rhs - sum(c * u_hat[v] for c, v in non_i_set)   # x since row_rhs aready has -
                self.mip.model.addVarToRow(row, z_var, coef_z)
                for c, v in i_set:
                    self.mip.model.addVarToRow(row, self.mip.vars[v], -c)

                self.mip.model.addCut(row)
                separated = True

        if separated:
            return {"result": SCIP_RESULT.SEPARATED}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


    def sepaexeclp(self):
        return self._check(None)

    def sepaexecsol(self, solution):
        return self._check(solution)