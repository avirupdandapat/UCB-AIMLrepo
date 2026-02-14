# UCB AIML Repository: Exploratory Data Analysis Projects

This repository contains two data analysis assignments implemented in Jupyter notebooks using Python.

## Repository Structure

```text
UCB-AIMLrepo/
├── assignment5_1/
│   ├── README.md
│   ├── data/
│   │   └── coupons.csv
│   ├── images/
│   │   ├── figure_01.png
│   │   ├── ...
│   │   └── figure_09.png
│   └── notebook/
│       └── prompt.ipynb
├── assignment6/
│   ├── titanic.csv
│   └── titanic_eda.ipynb
└── README.md
```

## Projects Included

### 1) Assignment 5.1: Coupon Acceptance Analysis
- Notebook: `assignment5_1/notebook/prompt.ipynb`
- Dataset: `assignment5_1/data/coupons.csv`
- Focus: Analyze factors that influence whether drivers accept coupons.

#### Code Workflow (Assignment 5.1)
1. Import libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `IPython.display`).
2. Load and audit data (`shape`, null checks, basic summaries).
3. Clean/normalize fields (notably handling nulls in `car` and related cleanup).
4. Compute global acceptance rate.
5. Visualize coupon categories and acceptance rates.
6. Deep-dive on **Bar** coupons with segment comparisons.
7. Independent deep-dive on **Coffee House** coupons by behavior/context.

#### Key Outputs Captured in Notebook
- Overall coupon acceptance rate: **56.93%**
- Bar coupon acceptance rate: **41.19%**
- Coffee coupon acceptance rate: **49.63%**
- Visuals exported in `assignment5_1/images/`.

### 2) Assignment 6: Titanic EDA
- Notebook: `assignment6/titanic_eda.ipynb`
- Dataset: `assignment6/titanic.csv`
- Focus: Exploratory analysis of survival patterns.

#### Code Workflow (Assignment 6)
1. Imports and setup.
2. Basic dataset overview and descriptive stats.
3. Missing-value analysis and handling.
4. Demographic plots (age/sex/class).
5. Feature engineering:
   - `FamilySize = Siblings/Spouses Aboard + Parents/Children Aboard + 1`
   - `IsAlone`
6. Survival-focused comparisons (boxplots, grouped rates, correlations).
7. Auto summary of findings.

#### Key Notebook Findings
- Survival by sex:
  - Female: `0.742`
  - Male: `0.190`
- Survival by passenger class:
  - Class 1: `0.630`
  - Class 2: `0.473`
  - Class 3: `0.244`
- Mid-size families show better survival than solo passengers.

## Important Libraries

The notebooks rely on:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `IPython` (`IPython.display` in assignment5_1)
- `jupyter` / `notebook` runtime

Install with:

```bash
pip install pandas numpy matplotlib seaborn jupyter ipython
```

## How To Run the Code

### Option A: Run interactively in Jupyter

```bash
# from repo root
jupyter notebook
```

Then open and run:
- `assignment5_1/notebook/prompt.ipynb`
- `assignment6/titanic_eda.ipynb`

### Option B: Run in VS Code Notebook UI
1. Open repo folder.
2. Select Python interpreter with required packages.
3. Open each notebook and run all cells.

## Data and Analysis Flow Diagrams

### High-level repository workflow

```mermaid
flowchart TD
    A[Raw CSV Data] --> B[Notebook EDA]
    B --> C[Cleaning and Feature Engineering]
    C --> D[Statistics and Segment Analysis]
    D --> E[Visualizations]
    E --> F[Insights and Conclusions]

    A1[assignment5_1/data/coupons.csv] --> B1[prompt.ipynb]
    A2[assignment6/titanic.csv] --> B2[titanic_eda.ipynb]
```

### Assignment 5.1 pipeline

```mermaid
flowchart LR
    A[coupons.csv] --> B[Audit and Null Analysis]
    B --> C[Data Cleaning]
    C --> D[Acceptance Rate Calculations]
    D --> E[Coupon Type Visualizations]
    E --> F[Bar Coupon Segmentation]
    F --> G[Coffee House Investigation]
    G --> H[Behavioral Insights]
```

### Assignment 6 pipeline

```mermaid
flowchart LR
    A[titanic.csv] --> B[Overview and Missing Analysis]
    B --> C[Cleaning]
    C --> D[Feature Engineering FamilySize IsAlone]
    D --> E[Demographic and Survival Plots]
    E --> F[Correlation and Group Summaries]
    F --> G[EDA Findings]
```

## Notes
- `assignment5_1/README.md` is project-specific and includes embedded figures.
- This root README summarizes both projects and how to execute them end-to-end.
