# Verification of Neural Networks

### Required Software
* Python 3.6+
* SCIP 6.0.0 or SCIP 6.0.1
* PySCIPopt

### Installation


### Execution
Chage into the folder src, i.e. `cd src`.  

Run `python3 execute.py` to execute a simple example. This will create a folder logs at the root of the repository. The configuration parameters can be changed directly in the file `execute.py`.

For normal execution, use the following arguments:  
```
python3 execute.py <path/to/test/instance> <path/for/logfile> <path/to/configuration/file> 
```  
The <path/to/test/instance> should point to an .rlv file as those under the folder benchmarks/scip. Please note that two log files of the form <path/for/logfile>.resultlog and <path/for/logfile>.runlog will be created. For example, you can run    
```
python3 execute.py ../benchmarks/scip/ACAS/property4/3_7.rlv ../log ../configs/no_heur_base.conf
```
