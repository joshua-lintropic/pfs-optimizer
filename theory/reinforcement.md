[neurips paper on robust learning (2000)](https://proceedings.neurips.cc/paper_files/paper/2000/file/e8dfff4676a47048d6f0c4ef899593dd-Paper.pdf)
- seems to be first of its kind, but limited b/c old
- analyzes control problems/physical simulations

[distributionally robust MDPs (2018)](https://arxiv.org/pdf/1801.04745)
- general formulation w/ uncertain transition probabilities, expected rewards, etc. 
- requires an "ambiguity set" by defining valid range of transition probabilities
- analyzes certain classes that are easier to prove results about

[robust policy evaluation (2024)](https://repository.gatech.edu/server/api/core/bitstreams/a601bd84-d4c5-4727-b94e-4d296d8a118e/content)
- ch 7 on first order policy optimization
- transforms into a robust optimization problem (i.e. sampling from training environment)
- online/hybrid robust MDP for potentially altered testing environments
