# Verification of Neural Networks

### Required Software
* Linux/Unix OS (we used Ubuntu and Fedora)
* Python 3.6+
* SCIP 6.0.0 or SCIP 6.0.1  (https://scip.zib.de)
* Our fork of PySCIPopt  (https://github.com/roessig/PySCIPOpt)

### Installation
At https://scip.zib.de/, download the SCIP optimization suite.
Use the hints at https://scip.zib.de/doc-6.0.1/html/CMAKE.php to install the SCIP. Please note that SCIP is available under the ZIB academic license (https://scip.zib.de/doc-6.0.1/html/LICENSE.php), which allows the free use of SCIP for academic, but not commercial purposes. You can use 
```
cmake .. -DCMAKE_INSTALL_PREFIX=/custom/scip/path
make install
```
to install SCIP to a custom path. Then you need to clone our fork of the PySCIPopt repository and run
```
(sudo) SCIPOPTDIR=/custom/scip/path python3 setup.py install
```
from the repository root directory. This should install the PySCIPopt fork to the corresponding Python installation. Of course, you can use a Python virtual environment instead. Further instructions can be found here https://github.com/SCIP-Interfaces/PySCIPOpt/blob/master/INSTALL.md.

### Execution
Change into the folder src, i.e. `cd src`.  

Run `python3 execute.py` to execute a simple example. This will create a folder logs at the root of the repository. The configuration parameters can be changed directly in the file `execute.py`.

For normal execution, use the following arguments:  
```
python3 execute.py <path/to/test/instance> <path/for/logfile> <path/to/configuration/file> 
```  
The <path/to/test/instance> should point to an .rlv file as those under the folder benchmarks/scip. Please note that two log files of the form <path/for/logfile>.resultlog and <path/for/logfile>.runlog will be created. For example, you can run    
```
python3 execute.py ../benchmarks/scip/ACAS/property4/3_7.rlv ../log ../configs/no_heur_base.conf
```
