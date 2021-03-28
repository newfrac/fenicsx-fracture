# NewFrac FEniCSx Training
## 30th March 2020 at Sorbonnes Université

This repository contains material for a day-long course on using DOLFINx,
the computational problem solving environment of the FEniCS Project. The
focus will be on solving problems arising in solid mechanics.

### Running the notebooks (to be tested *prior* to course start)

Students should execute the notebooks using a Docker container running on
their own computers.

1. Install Docker following the instructions at
   https://www.docker.com/products/docker-desktop.

2. Clone this repository using git:

       git clone git@github.com:jhale/newfrac-fenicsx-training.git

3. Run `./launch-notebook.sh`.

4. You should be able to see the JupyterLab instance by navigating to
   https://localhost:8888 in your web browser.

5. The JupyterLab session will ask you for a token (password). This can be
   found in the output from the terminal and will look like e.g.
   `b64972b8b7df3717089c4899bd028f5e2df6a73a845cb250`.

Although we recommend using Docker locally, you can also use the cloud-based
binder service to execute the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jhale/newfrac-fenicsx-training/HEAD)

### Prerequisite knowledge

The course will assume basic knowledge of the theory of finite elasticity and
finite element methods. We will also assume that the students have taken the
NEWFRAC Core School 2021 *Basic computational methods for fracture mechanics*.
Basic knowledge of Python will be assumed, see https://github.com/jakevdp/WhirlwindTourOfPython
to brush up if you feel unsure.

### Course Schedule

* 0900-1030 Session 1 (1hr30m). Introduction to DOLFINX and linear elasticity.
* 1030-1045 Break (15m).
* 1045-1215 Session 2 (1hr30m). Finite elasticity 
* 1215-1315 Lunch (1hr).
* 1315-1445 Session 3 (1hr30m). Solver for Variational Inequalities. Variational theory of fracture.
* 1445-1500 Break. (15m).
* 1500-1630 Session 4 (1hr30m). Variational theory of fracture.

### Capstone Project

### Instructors/Authors

Jack S. Hale, University of Luxembourg.
Corrado Maurini, Sorbonnes Université.

### Acknowledgements

The funding received from the European Union’s Horizon 2020 research and
innovation programme under Marie Skłodowska-Curie grant agreement No.
861061-NEWFRAC is gratefully acknowledged.

### License

MIT License, see `LICENSE` file.
