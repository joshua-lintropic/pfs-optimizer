# Program Usage
## Creating New Tests
To get started, navigate to `./tests` and make a new directory with the name of your test case (no spaces). Then navigate back to the root and open `tester.py`, and add your new test case function. The following are to be supplied as return values:
1) `N`, the number of galaxies to be selected from. 
2) `K`, the total number of fibers available per exposure.
3) `L`, the total number of exposures available. 
4) `T_exp`, the duration of 1 per exposure. 
5) `u_max`, an `ndarray` of size `(N,)` containing maximum utilities of observing a galaxy.
6) `T_target`, an `ndarray` of size `(N,)` containing the required target times to 

To view the distributions of `u_max` and `T_target` generated, execute `python3 tester.py testname`. The histograms will be located under `tests/testname/hist_umax.png` and `tests/testname/hist_target.png.` #todo 

## Running the Optimizer
To run the optimizer, execute `python3 optimizer.py testname start stop` where `start` and `stop` are optional. (If you supply just one value, it will be interpreted as `start). 
- If you do not supply `start` or `stop`, the optimizer will run all exposures `[1, L]`. 
- If you supply `start`, the optimizer will run exposures `[start, L]`. 
- If you supply `start` and `stop`, the optimizer will run exposures `[start, stop]`. 

## Loading Saved Results
If you choose to break up run of the optimizer, the program will by default store its results in `t.npy` and `obs.npy` to be continued later. 
- `t.npy` contains an `ndarray` of shape `(N,L)` which stores integer values `0` or `1` representing whether galaxy `i` was allocated a fiber in exposure `l`. Initialized to zeros. 
- `obs.npy` contains an `ndarray` of shape `(L,K)` which for each exposure `l` stores the `K` galaxies selected by the optimizer. Initialized to zeros. 

Naturally if `t.npy` and `obs.npy` are modified and then reloaded into the program, it will accept the new values gracefully. 

## Viewing Outputs
The primary text-based description of a run of the optimizer may be found in `log.txt`. Test cases designed to evaluate performance of the optimizer also come with `desc.txt` describing their purpose. 

Graphical files denoted `uhistxx.png` and `thistxx.png` are histograms of the maximum utility and target time for selected galaxies as the optimizer progresses, respectively. Note that if preloading occurs, **the optimizer will overwrite old images with new ones** to establish the progression of the histograms over time. 

Finally, `sharp.png` is a time series plot of the attained sharp utility over time. (The sharp utility is defined as only being attained when the entire target time for a galaxy is reached. It is all-or-nothing.)

# Problem Specification

See problem_specification.pdf for a description of the problem and an analysis of the approach that this optimizer takes.
