# AI-310 Course Projects

<img width="130" height="145" alt="capital-university" src="https://github.com/user-attachments/assets/9a940f0b-d3d3-454e-9413-dd7fd88d9f2e" align="right" />

**Faculty of Computing and Artificial Intelligence**
**Capital University** *~(Formerly Helwan University)*

**Course:** Intro. to Artificial Intelligence (AI-310)  
**Instructor:** Prof. Amr S. Ghoniem
**Academic Year:** 2026/2027

<br>


---

## Project Overview

The **Job Scheduling Problem (JSP)** is a classic NP-hard combinatorial optimization challenge where a set of jobs must be processed on available resources (machines). Each job consists of a strict sequence of operations, and each operation requires a specific machine for a predefined processing time. The primary objective of this project is to find a valid schedule that minimizes the **makespan** (total completion time) while respecting all precedence and machine capacity constraints.

To solve and analyze this problem, our system implements and benchmarks three distinct algorithmic approaches:

1. **Backtracking Search Algorithm:** An exact solver that systematically and recursively explores the decision space. It incrementally builds a schedule and backtracks upon constraint violation, guaranteeing the optimal solution for small problem instances.
2. **Genetic Algorithm (GA):** An evolutionary metaheuristic that maintains a population of candidate schedules. It utilizes specialized chromosome encoding, selection, crossover, and mutation operators to evolve and improve solutions over successive generations.
3. **Memetic Algorithm (MA):** A hybrid metaheuristic that extends the GA by integrating global population-based exploration with localized heuristic refinement (Individual Learning / Local Search). This ensures faster convergence and better escapes from local optima.

---

## Documentation

* **Official Guidelines:** The architecture, constraints handling, and algorithmic objectives are developed in accordance with the `job_scheduling_problem_solver.pdf` specification provided in the `guidelines/` directory[cite: 1, 2, 3, 4].
* **Benchmark Datasets:** The algorithms are evaluated against standard **OR-Library** job-shop instances (such as Fisher and Thompson's `ft06`, `ft10`, and Lawrence's `la01`). These are located in the `datasets/` directory.

> **Note:** Full docs are available in the `docs/` directory, including detailed explanations of the algorithms, their implementations, and performance analyses.
---

## Team Members & Responsibilities

| Name                  | GitHub                                                             | Module |
|-----------------------|--------------------------------------------------------------------|--------|
| Abdelhalim Yasser     | [@abdelhalimyasser](https://github.com/abdelhalimyasser)           | - Genetic Module |
| Ali Samy              | [@AliSamy12](https://github.com/Aly-Sami)                          | - Genetic Module |
| Nour Eldeen Shaban    | [@noureldeenshaaban](https://github.com/noureldeenshaaban)         | - Backtracking Module |
| Yousef Mohamed        | [@yousefmass](https://github.com/yousefmass)                       | - Backtracking Module |
| Norhan Ezzat          | [@nourhan-eg](https://github.com/nourhan-eg)                       | - Memetic Module |
| Mariam Mohamed        | [@MaRIam-Mo7amed](https://github.com/MaRIam-Mo7amed)               | - Memetic Module |

---

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/abdelhalimyasser/ai_genatic_memetic_backtracking_job_scheduling_problem_solver.git
cd ai_genatic_memetic_backtracking_job_scheduling_problem_solver
```
2. Do your work then
```bash
git add .
git commit -m "add your message's" -m "Co-authored-by: `git config user.name` <`git config user.email`>"
```

3. Push your changes
```bash
git push origin main
```

4. Pull the latest changes for the other members  *do this before starting your work*
```bash
git pull origin main
```

5. setup your virtual environment and install the requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## References

---
<p align="center">
  <strong>FCAI – Capital University ~ (Formerly Helwan University) </strong><br>
  Intro. to Artificial Intelligence · AI-310 · Final Project · 2026/2027<br>
</p>
