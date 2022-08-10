This repository contains the code used for the experiments of the PhD dissertation by Ilnura Usmanova.

- The results of the experiments we made on our machine are saved in /runs folder. 
- The corresponding convergence plots lie in /plots folder. 
- The corresponding trajectories plots lie in /evals folder. 
- The code responsible for the LB_SGD, for plotting the plots, and for running the SafeOpt lies in /libs folder.

To reproduce the experiments,

1. Open the file /evals/ExperimentsThesisSections5152.ipynb.ipynb. 
2. Install a Python 3 environment and missing packages if needed using command: 
    pip install {library name} 
    (one can do that directly in Jupyter Notebook in a separate cell)
3. Run the cells corresponding to the LB_SGD experimens (they will rewrite the results in /runs folder)
4. Same for the SafeOpt experimens if you want (running them takes longer time)
5. Run the cells corresponding to the plotting the experiments (this will use the information saved in /runs folder, and save the plot in /plots folder)

