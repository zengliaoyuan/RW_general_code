# RW_general_code


### TableGen_prs_3half.m, TableGen_prs_cubic.m, TableGen_stp_p3_s10.m
These three are runcodes.



### RW_prs.m, RW_stp
These two are implementation of Rendl-Wolkowicz algorithm for p-regularization subproblem and p-TRS



### Newton_prs.m, Newton_stp.m
These two are implementation of Newton method for p-regularization subproblem and p-TRS



### gep.m, minresQLP.m
gep.m is our implementation of the generalized-eigenvalue-based approach for cubic-regularization subproblem proposed in [Lieder, 2020](https://epubs.siam.org/doi/abs/10.1137/19M1291388). 

minresQLP.m is a subroutine used in gep.m (as suggested in [Lieder, 2020](https://epubs.siam.org/doi/abs/10.1137/19M1291388)), and this code is downloaded from https://web.stanford.edu/group/SOL/software/minresqlp/

# Reference

More details on implementation and numerical experience can be found in ["![rho](https://render.githubusercontent.com/render/math?math=\rho)-regularization subproblems: Strong duality and an eigensolver-based algorithm"](https://arxiv.org/abs/2109.01829)

