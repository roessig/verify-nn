from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
import logging


class DualBoundEvent(Eventhdlr):
    """This class is used to stop the solving process when an instance is proved to be verifiable or refutable."""

    def __init__(self, mip):
        self.mip = mip
        self.log = logging.getLogger('main_log')


    def eventexec(self, event):
        """Execute the event listener."""

        node = event.getNode()
        if event.getType == SCIP_EVENTTYPE.NODEINFEASIBLE:
            self.log.debug("Node %i infeasible, addedcons %i", node.getNumber(), node.getNAddedConss())

        if self.mip.model.getDualbound() > self.mip.eps:
            self.mip.model.interruptSolve()
            self.log.info("Interrupted solving, dual bound %f > 0.", self.mip.model.getDualbound())
            self.log.info("total time: %f", self.mip.model.getTotalTime())

        if self.mip.model.getPrimalbound() < -self.mip.eps:
            self.mip.model.interruptSolve()
            self.log.info("Interrupted solving, primal bound %f < 0.", self.mip.model.getPrimalbound())
            self.log.info("total time: %f", self.mip.model.getTotalTime())

