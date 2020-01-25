# Verification of Neural Networks

This repository contains the code for my solver for verification of neural networks as described in my Master's thesis with the same title which can be found [here](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7417) (urn:nbn:de:0297-zib-74174). Furthermore, in the folder benchmarks are the test instances which were used for the computational study in the thesis. This includes the test instances for
* [Neurify](https://github.com/tcwangshiqi-columbia/Neurify)
* [BaB](https://github.com/oval-group/PLNN-verification/tree/newVersion) of Bunel et al.
* [Reluplex](https://github.com/guykatzz/ReluplexCav2017) (for the computational experiments the [fork](https://github.com/bunelr/ReluplexCav2017/tree/e316ae8193dfe6d6f8b869e95a8502c5316b3d87) of Bunel et al. was used)

and of course the test instances for our solving model based on the academic MIP solver [SCIP](scip.zib.de). The file format of our .rlv input files is closely oriented at the format of Ehlers used for the solver [Planet](https://github.com/progirep/planet). If you want to define your own instances, please see the file [DATAFORMAT.md](https://github.com/roessig/verify-nn/blob/master/DATAFORMAT.md). 

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
from the repository root directory. This should install the PySCIPopt [fork](https://github.com/roessig/PySCIPOpt) to the corresponding Python installation. Of course, you can use a Python virtual environment instead. Further instructions can be found here https://github.com/SCIP-Interfaces/PySCIPOpt/blob/master/INSTALL.md.

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
