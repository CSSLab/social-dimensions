# social-dimensions

Code to reproduce the social dimensions and analyses from the 2021 paper ["Quantifying social organization and political polarization in online platforms"](https://doi.org/10.1038/s41586-021-04167-x) by Isaac Waller and Ashton Anderson.

## Requirements

* Python 3.x
* Spark and `pyspark`
* `pandas`
* Software that can run Jupyter notebooks

## Instructions to reproduce social dimensions

1. Load the `social-dimensions.ipynb` notebook.
2. Run all cells in the notebook.
3. Resulting scores for all communities will be saved in the `scores.csv` file, as well as the `scores` Pandas DataFrame in the notebook for you to explore.

See `scores.csv` from the repository for full example output, which this code should reproduce exactly.

## Instructions to reproduce analyses / plots from paper

1. You will need to first download the Pushshift data (see script `commembed/data/download.sh`) and then import it to parquet format (see script `commembed/data/import_data.py`).
2. Notebooks to generate all the plots are in the `notebooks` folder. They are ordered because some notebooks generate data that later notebooks depend on.

## Citation

If you use any data or code from this repository, please cite our paper:

> Waller, I., Anderson, A. Quantifying social organization and political polarization in online platforms. Nature 600, todo-todo (2021). https://doi.org/10.1038/s41586-021-04167-x

## Contact

If you have any questions, please [contact us](http://csslab.cs.toronto.edu/people/).