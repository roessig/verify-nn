from model_boundd import MIPwithBounds
from check_solution import check_sol
import logging
from os import listdir, path, makedirs
import datetime
import sys
import ast


if __name__ == "__main__":

    # read in input file and configuration, set up logging directories
    if len(sys.argv) == 4:
        filepath = sys.argv[1]
        run_logs_folder = path.dirname(sys.argv[2])
        with open(sys.argv[3], "r") as f:
            s = f.read()
            params = ast.literal_eval(s)
        result_log_path = sys.argv[2] + ".resultlog"
        run_log_filename = path.basename(sys.argv[2]) + ".runlog"
    
    # if arguments are not provided, run an example file and use paramters as below
    else:
        makedirs("../logs", exist_ok=True)
        run_logs_folder = "../logs"
        result_log_path = "../logs/results.log"
        filepath = "rlv_test"

        #####################
        # General Parameters
        #####################

        params = {}
        params["eps"] = 5*1e-8
        params["timelimit"] = 10000
        params["presolving_rounds"] = 0
        params["use_opt_mode"] = True
        params["build_optimize_nodes"] = False
        params["build_use_symbolic"] = False
        params["use_linear_model"] = True
        params["delete_linear_cons"] = True
        params["sampling_heuristic_local_max_iter"] = 1000   # use negative value to disable
        params["sampling_heuristic_local_freq"] = 1
        params["sampling_heuristic_local_maxdepth"] = 20

        params["sampling_heuristic_max_iter"] = 1000        # use negative value to disable
        params["sampling_heuristic_freq"] = 1
        params["sampling_heuristic_maxdepth"] = 0
        params["sampling_heuristic_bound_for_lp_heur"] = 100000.0  # use negative value to disable LP heur
        params["sampling_heuristic_max_iter_lp_heur"] = 5000
        params["sampling_heuristic_use_lp_sol_gen"] = True

        params["use_domain_branching"] = False
        params["domain_branching_split_mode"] = "gradient"
        params["domain_branching_priority"] = 100000
        params["domain_branching_maxdepth"] = 20
        params["domain_branching_maxbounddist"] = 1

        params["use_relu_branching"] = True
        params["relu_branching_split_mode"] = ""
        params["relu_branching_priority"] = 100000
        params["relu_branching_maxdepth"] = 20
        params["relu_branching_maxbounddist"] = 1

        params["use_obbt_propagator"] = True
        params["obbt_maxdepth"] = 20
        params["obbt_use_genvbounds"] = False
        params["use_obbt_two_variables"] = False
        params["obbt_optimize_nodes"] = True
        params["obbt_use_symbolic"] = False
        params["obbt_bound_for_opt"] = -200           # use negative value to disable

        params["bfs_from_all_inputs"] = True

        params["use_ideal_separator"] = False
        params["sepa_freq"] = 5
        params["sepa_priority"] = 100
        params["sepa_maxbounddist"] = 0.0
        params["sepa_delay"] = False


        now = datetime.datetime.now()
        run_log_filename = path.basename(path.dirname(filepath)) + "_" + path.basename(filepath) + \
                           "-".join((str(x).zfill(2) for x in
                                     (now.year, now.month, now.day, now.hour, now.minute))) + ".runlog"

        # end if for manual mode

    logger = logging.getLogger('main_log')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug

    handler = logging.FileHandler(path.join(run_logs_folder, run_log_filename), mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    mip = MIPwithBounds(filepath, params["eps"])
    logger.info("call read_file_into_graph with file %s", filepath)
    model, vars = mip.read_file_into_graph()
    model.printVersion()

    model.setIntParam("presolving/maxrounds", params["presolving_rounds"])
    model.setBoolParam("presolving/donotmultaggr", True)
    model.setBoolParam("presolving/donotaggr", True)
    model.setIntParam("nodeselection/bfs/stdpriority", 9999999)   # bfs is best first search
    model.setIntParam("nodeselection/bfs/memsavepriority", 9999999)
    model.setIntParam("nodeselection/bfs/minplungedepth", 0)
    model.setIntParam("nodeselection/bfs/maxplungedepth", 0)
    model.setRealParam("nodeselection/bfs/maxplungequot", 0.0)
    model.setIntParam("nodeselection/breadthfirst/stdpriority", 0)
    model.setIntParam("nodeselection/breadthfirst/memsavepriority", 0)
    model.setIntParam("nodeselection/dfs/stdpriority", -10000)
    model.setIntParam("display/freq", 2)
    model.setIntParam("display/headerfreq", 2)
    model.setIntParam("presolving/maxrestarts", 0)
    model.setLongintParam("lp/rootiterlim", -1)
    model.setIntParam("lp/solvefreq", 1)    # 0 = only at root node, else frequency of KP solves
    model.setIntParam("separating/flowcover/freq", -1)
    model.setIntParam("separating/gomory/freq", -1)
    model.setIntParam("separating/cmir/freq", -1)
    model.setIntParam("propagating/maxroundsroot", 5)
    model.setIntParam("propagating/maxrounds", 1)
    model.setIntParam("display/verblevel", 5)


    mip.add_further_constraints(optimize_nodes=params["build_optimize_nodes"],
                                linear_model=params["use_linear_model"],
                                opt_mode=params["use_opt_mode"],
                                use_symbolic=params["build_use_symbolic"],
                                bfs_from_all_inputs=params["bfs_from_all_inputs"])

    mip.add_binary_constraints(delete_cons=params["delete_linear_cons"])
    print(filepath)

    mip.add_optimize_constraints(params["use_opt_mode"])
    mip.build_pytorch_model()

    if params["sampling_heuristic_max_iter"] >= 0:
        mip.add_sampling_heuristic(max_iter=params["sampling_heuristic_max_iter"],
                                   opt_mode=params["use_opt_mode"],
                                   bound_for_lp_heur=params["sampling_heuristic_bound_for_lp_heur"],
                                   max_iter_lp_heur=params["sampling_heuristic_max_iter_lp_heur"],
                                   freq=params["sampling_heuristic_freq"],
                                   maxdepth=params["sampling_heuristic_maxdepth"],
                                   use_lp_sol_gen=params["sampling_heuristic_use_lp_sol_gen"],
                                   use_relu_branch_gradient=True if params["relu_branching_split_mode"] ==
                                                                    "gradient" else False)

    if params["sampling_heuristic_local_max_iter"] >= 0:
        mip.add_sampling_heuristic_local(max_iter=params["sampling_heuristic_local_max_iter"],
                                         opt_mode=params["use_opt_mode"],
                                         bound_for_lp_heur=params["sampling_heuristic_bound_for_lp_heur"],
                                         max_iter_lp_heur=params["sampling_heuristic_max_iter_lp_heur"],
                                         freq=params["sampling_heuristic_local_freq"],
                                         maxdepth=params["sampling_heuristic_local_maxdepth"],
                                         use_lp_sol_gen=params["sampling_heuristic_use_lp_sol_gen"])

    if params["use_domain_branching"]:
        mip.add_domain_branching(opt_mode=params["use_opt_mode"],
                                 split_mode=params["domain_branching_split_mode"],
                                 priority=params["domain_branching_priority"],
                                 maxdepth=params["domain_branching_maxdepth"],
                                 maxbounddist=params["domain_branching_maxbounddist"])

    if params["use_relu_branching"]:
        mip.add_relu_branching(priority=params["relu_branching_priority"],
                               maxdepth=params["relu_branching_maxdepth"],
                               maxbounddist=params["relu_branching_maxbounddist"])

    if params["use_obbt_propagator"]:
        prop = mip.add_dnn_bound_prop(opt_mode=params["use_opt_mode"],
                                  optimize_nodes=params["obbt_optimize_nodes"],
                                      obbt_2=params["use_obbt_two_variables"],
                                      use_symbolic=params["obbt_use_symbolic"],
                                      bound_for_opt=params["obbt_bound_for_opt"],
                                      maxdepth=params["obbt_maxdepth"],
                                      use_genvbounds=params["obbt_use_genvbounds"],
                                      **{"obbt_k": params["obbt_k"],
                                         "obbt_l": params["obbt_l"],
                                         "obbt_sort": params["obbt_sort"]} if params["use_obbt_two_variables"] else {})

    if params["use_ideal_separator"]:
        mip.add_relu_sepa(priority=params["sepa_priority"],
                          freq=params["sepa_freq"],
                          maxbounddist=params["sepa_maxbounddist"],
                          delay=params["sepa_delay"])


    mip.add_eventhdlr_dualbound()
    model.presolve()
    mip.catch_events()


    with open(result_log_path, "a") as f:
        f.write("\ndate_time: " + str(datetime.datetime.now()) + "\n")
        f.write("filepath: " + filepath + "\n")
        f.write("\n".join(x + ": " + str(y) for x, y in params.items()) + "\n")


    model.setRealParam("limits/time", params["timelimit"])

    logger.info("start optimizing, total time is %f", model.getTotalTime())
    model.optimize()

    print(model.getStatus())
    print("Total Time", model.getTotalTime())
    # model.printStatistics()         # prints statistics about the solving process

    sol_is_valid = False
    if (not params["use_opt_mode"] and model.getStatus() == "optimal") or (params["use_opt_mode"]
                                        and model.getPrimalbound() <= -mip.eps):
        print("SAT")
        print("cheking solution")
        input_values = {var.name: model.getVal(var) for var in mip.input_nodes.values()}
        print(input_values)
        sol_is_valid = check_sol(filepath, input_values)
        print("is valid:", sol_is_valid)

    if (not params["use_opt_mode"] and model.getStatus() == "infeasible") or (params["use_opt_mode"]
                                        and model.getDualbound() >= mip.eps):
        print("UNSAT")

    with open(result_log_path, "a") as f:
        f.write("solution_status: " + str(model.getStatus()) + "\n")
        f.write("total_time: " + str(model.getTotalTime()) + "\n")
        f.write("primal_bound: " + str(model.getPrimalbound()) + "\n")
        f.write("dual_bound: " + str(model.getDualbound()) + "\n")
        f.write("number_of_nodes: " + str(model.getNNodes()) + "\n")
        f.write("positive_solution_valid: " + str(sol_is_valid) + "\n")

