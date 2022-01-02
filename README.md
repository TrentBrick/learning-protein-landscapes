
Supervised by Dr. Debora Marks's Lab as my undergrad senior thesis - This research was motivated by the promise of recent developments in our ability to predict protein functionality and the problem of finding novel sequences that maximize this prediction. We tried developing a new solution using invertible neural networks and variational inference to approximate the intractable distribution of any protein function predictor with reason to believe it would outperform Markov Chain Monte Carlo methods.

This codebase was used to run all experiments presented in my thesis defense which is attached as `Learning_Fitness_Landscapes_for_Protein_Design_TrentonBricken.pdf`.

Ultimately, the project was unsuccessful as it proved too difficult to get the normalizing flows to learn useful protein representations. However, working with state of the art variational inference methods was exciting. Developing baselines like the hill climbing algorithm was also interesting.

This work was presented at the NeurIPS 2019 workshop, Learning Meaningful Representations of Life (LMRL).

# Summary of Codebase

* NoeFolders contains code from the excellent paper: ["Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning"](https://www.science.org/doi/10.1126/science.aaw1147) that this project used as its original foundation. 

* notebooks - contain code to debug and analyze results from the models

* tensorf - contains modified code from the NoeFolders paper, it needed to be modified to run correctly and establish benchmarks

* pytorch - pytorch equivalent code to the tensorflow variants. Includes a large number of normalizing flow variants such as bipartite, MADE, Spline flows, and Discrete flows. Also scripts to work with protein sequences and run hill climbing optimization.

# Acknowledgements

Thanks to Dr. Debora Marks, Dr. Robert Thompson and Dr. David Banks for supervising the thesis defense. 

Thanks to Nikki Thani and Nathan Rollins for their involvement in the research and to the rest of the Marks lab too. 