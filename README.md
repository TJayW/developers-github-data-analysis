# GitHub Data Analysis

This repository contains two main scripts in Python:

1. **github_data_collector.py**  
   - **Description**:  
     Collects data from GitHub (repositories, contributors, collaborations, pull requests) using the GitHub API ([PyGithub](https://github.com/PyGithub/PyGithub) library).
   - **Output**:  
     Generates various CSV files:
     - `developers_large.csv` (developers details)
     - `repositories_large.csv` (repository details)
     - `collaborations_large.csv` (collaborations, reviews/comments on PR)
     - `pull_requests_large.csv` (details of PRs)
   - **Execution**:
     ```bash
     # Set the environment variable with your GitHub token
     export GITHUB_TOKEN=your_personal_token

     # Start the script
     python github_data_collector.py
     ```

2. **github_data_analysis.py**  
   - **Description**:  
     Analyzes the data collected by the first script, producing network metrics, repository and pull request statistics, and visualizations (graphs and CSV files).
   - **Output**:
     - `developer_metrics_analysis.csv` and `developer_metrics_analysis_sorted.csv`  
       (centrality and community metrics for each developer).
     - `repository_statistics.csv`  
       (descriptive statistics for repositories).
     - `pull_requests_statistics.csv`  
       (descriptive statistics for PRs).
     - `network_statistics.csv` and `network_statistics_sorted.csv`  
       (network statistics such as density, transitivity, etc.).
     - Various graphs in `.png` format (distributions, scatter plots, communities, etc.).
   - **Execution**:
     ```bash
     python github_data_analysis.py
     ```

## Requirements

- **Python 3.8+** (3.9 or 3.10 recommended).
- [PyGithub](https://github.com/PyGithub/PyGithub)
- [pandas](https://pandas.pydata.org/)
- [networkx](https://networkx.org/)
- [matplotlib](https://matplotlib.org/)
- [community-louvain (python-louvain)](https://pypi.org/project/python-louvain/)
- [tqdm](https://pypi.org/project/tqdm/)

Install the dependencies with:
```bash
pip install -r requirements.txt