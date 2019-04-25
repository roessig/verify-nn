from pyscipopt import Branchrule, SCIP_RESULT
import logging


class ReluBranching(Branchrule):
    """Class which implements the ReLU branching rule. Sampling heuristic must run up to the same depth as this
    branching rule, otherwise this branching rule does not work, since sampling heuristic fills self.mip.
    nodes_by_branch_prio."""


    def __init__(self, mip):

        self.mip = mip
        self.was_branched = set()
        self.log = logging.getLogger('main_log')


    def branchexeclp(self, allowaddcons):
        """Execute ReLU branching based on LP solution."""

        assert allowaddcons
        vars, sols, fracs, nlpcands, nprio, nfrac = self.mip.model.getLPBranchCands()
        var_names = set(v.name for v in vars[:nprio])
        for i, node_name in enumerate(self.mip.nodes_by_branch_prio[self.mip.model.getCurrentNode().getNumber()]):

            if node_name in self.mip.relu_nodes and "t_bin_" + node_name in var_names:

                down, eq, up = self.mip.model.branchVar(self.mip.binary_variables["bin_" + node_name])
                print("branched at node", self.mip.model.getCurrentNode().getNumber(), "on", node_name, self.mip.vars[node_name + "_in"].getLbLocal(),
                     self.mip.vars[node_name + "_in"].getUbLocal())
                self.log.debug("relu branched on %s", node_name)
                self.mip.nodes_by_branch_prio[self.mip.model.getCurrentNode().getNumber()][i] = None
                self.mip.model.addCons(self.mip.vars[node_name + "_in"] <= 0.0, node=down, local=True)
                self.mip.model.addCons(self.mip.vars[node_name + "_in"] == self.mip.vars[node_name], node=up, local=True)
                self.mip.model.chgVarUbNode(down, self.mip.vars[node_name + "_in"], 0.0)   # in <= 0
                self.mip.model.chgVarUbNode(down, self.mip.vars[node_name], 0.0)   # out <= 0
                self.mip.model.chgVarLbNode(up, self.mip.vars[node_name + "_in"], 0.0)     # in >= 0, not necessary for out

                return {"result": SCIP_RESULT.BRANCHED}

        self.log.debug("relu branching could not branch")
        return {"result": SCIP_RESULT.DIDNOTRUN}


    def branchexecext(self, allowaddcons):
        print("Relu branch ext", allowaddcons)
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def branchexecps(self, allowaddcons):
        print("Relu branch not all fixed", allowaddcons)
        return {"result": SCIP_RESULT.DIDNOTRUN}

