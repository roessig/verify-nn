from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

class LbChangeEvent(Eventhdlr):
    """Event handler for bound changes. Can be used to debug strange variable bound changes."""

    def eventexec(self, event):
        print("ub event")
        #if event.getType == SCIP_EVENTTYPE.VARCHANGED:
            #if event.getVar().name == "relu_0X24":
        print("transformed variable", event.getVar().name)
        print("oldbound:", event.getOldBound())
        print("newbound:", event.getNewBound())

