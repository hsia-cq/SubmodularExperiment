# Influence_Maximization_Experiment

To run the Influence Maximization Experiment, follow these steps:

1. **Download all files** (both data and code) in the `Influence_Maximization_Experiment` folder.
2. **Open** `Influence_Maximization_Experiment.ipynb` in Jupyter Notebook.
3. **Run the notebook** from top to bottom.

The code includes four sub-experiments:

1. `Influence_Comparison_with_RSOP`
2. `Visualise_Influence_Maximization_Result`
3. `Comparison_with_increasing_error`
4. `Influence_Comparison_Large_Scale_Network`

## ML Feature Selection Experiment

There are seven Python files and two datasets for this experiment. You need to download the datasets `airline.csv` and `breast-cancer.csv` and follow these steps:

1. For the SOP-Greedy and benchmark comparison, you can directly obtain the plots by running `airline_performance.py` and `breast_cancer_performance.py`.
2. For the various eta versus accuracy figures, run `airline_eta.py` and `cancer_eta.py`.
3. For the r-step bar charts:
    - Adjust the `max_K` values in `airline_rstep.py` and `cancer_rstep.py` from 3 to 7.
    - Run these files to get the values.
    - Use the obtained values in `RSTEP_plot.py` to generate the output plot.

## Text Summarization Experiment

This experiment involves six data files and five Python files. You need to download the six BBC text files, which include the news text and the reference summary text for each data category. Follow these steps:

1. For generating bar charts for different K values:
    - Run `coverage.py`, `diversity.py`, and `facility_location.py` by entering K values from 3 to 7.
    - Obtain the median and IQR for each of the three objective functions.
    - Paste the obtained values into `plot_different_k.py` to generate the plot.

2. For generating bar charts for different news categories:
    - Run the three function files, varying the constraint number K from 3 to 5.
    - Obtain the median and IQR for different functions under the news datasets.
    - Paste the obtained values into `plot_different_news.py` to generate the plot.
