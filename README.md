# turbine_opt_adapt

The experimental configurations provided in this repo demonstrate the
combination of PDE-constrained optimisation methods with goal-oriented mesh
adaptation. All experiments are conducted in
[Thetis](https://thetisproject.org) - a discontinuous Galerkin coastal ocean
model built on top of [Firedrake](https://firedrakeproject.org). The
goal-oriented mesh adaptation functionality is provided by
[Goalie](https://github.com/mesh-adaptation/goalie) and
[Animate](https://github.com/mesh-adaptation/animate).

## Installation

1. Install PETSc with the Mmg metric-based mesh adaptation tool by following the
   instructions provided in the
   [mesh-adaptation wiki](https://github.com/mesh-adaptation/docs/wiki/Installation-Instructions).
2. Install Firedrake on top of the PETSc build from step 1 by following the
   instructions provided in the
   [mesh-adaptation wiki](https://github.com/mesh-adaptation/docs/wiki/Installation-Instructions).
3. Install Animate and Goalie into the same virtual environment used for
   installing Firedrake by following the instructions provided in the
   [mesh-adaptation wiki](https://github.com/mesh-adaptation/docs/wiki/Installation-Instructions).
4. Clone this repository and install it into the same virtual environment used
   above:
```sh
source /path/to/venv/bin/activate
git clone https://github.com/mesh-adaptation/turbine_opt_adapt.git
cd turbine_opt_adapt
pip install .
```
