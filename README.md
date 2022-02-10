# Reddit social dimensions

Data and code for the community embedding, social dimensions, and analyses from the 2021 paper ["Quantifying social organization and political polarization in online platforms"](https://doi.org/10.1038/s41586-021-04167-x) by Isaac Waller and Ashton Anderson.

## Data

The following data has been made available in the `data` directory:

### Community embedding

The community embedding of Reddit used in the paper. `embedding-vectors.tsv` contains the 150-dimensional vectors for each community, while `embedding-metadata.tsv` contains the name, description and associated data for each community. Communities with similar user bases are similar in the embedding; see Methods, Creating the community embedding.

`embedding-vectors.tsv`:
```
0.019756        -0.07609199999999999    -0.017321       -0.024236       0.112748        -0.10828099999999999    -0.35062        0.401909        -0.254341  0.260575 0.204183    ...
-0.004445       -0.036706       -0.019637       0.129492        -0.045198       -0.067518       0.07739700000000001     0.16213 -0.022069       0.060171   0.34275100000000003      0.032792        -0.124957       0.114371        ...
...
```

`embedding-metadata.tsv`:
```
community       description     over18
keto    The Ketogenic Diet is a low carb, high fat method of eating. And /r/keto is place to share thoughts, ideas, benefits, and experiences around eating within a Ketogenic Diet...     False
AskReddit       /r/AskReddit is the place to ask and answer thought-provoking questions.        False
...
```

### Social dimensions
The communities used to construct each social dimension are listed in `social-dimensions.yaml`. The first pair was manually provided while the rest were automatically found as per Methods, Finding social dimensions.

`social-dimensions.yaml`:
```
dimensions:
	- name: age
	  seeds:
		- [teenagers, RedditForGrownups]
		- [youngatheists, TrueAtheism]
....
```

### Social dimension scores
The scores for all of the 10,000 Reddit communities on each of our social dimensions (ex. `age`, `partisan`) and associated `neutral` dimensions are available in `scores.csv`. 

`scores.csv`:
```
community,age,gender,partisan,...
keto,0.17760505920402261,0.10308876095697105,-0.015496712806190574,...
AskReddit,-0.07415413657149496,0.13052107711645367,0.05281928294403579,...
...
```

### Figure data
The underlying data for all main text figures from the paper are available in `data/figure_data`.

## Citation

If you use any data or code from this repository, please cite our paper:

Waller, I., Anderson, A. Quantifying social organization and political polarization in online platforms. *Nature* 600, 264–268 (2021). https://doi.org/10.1038/s41586-021-04167-x

## Reproduction code

Code to reproduce the analyses from the paper is available in `full_code/`.

### Requirements

* Python 3.x
* Spark and `pyspark`
* `pandas`
* Software that can run Jupyter notebooks

### Instructions to reproduce social dimensions

1. Load the `full_code/social-dimensions.ipynb` notebook.
2. Run all cells in the notebook.
3. Resulting scores for all communities will be saved in the `scores.csv` file, as well as the `scores` Pandas DataFrame in the notebook for you to explore.

See `full_code/scores.csv` from the repository for full example output, which this code should reproduce exactly.

### Instructions to reproduce analyses / plots from paper

1. You will need to first download the Pushshift data (see script `full_code/commembed/data/download.sh`) and then import it to parquet format (see script `full_code/commembed/data/import_data.py`).
2. Notebooks to generate all the plots are in the `full_code/notebooks` folder. They are ordered because some notebooks generate data that later notebooks depend on.

## Contact

If you have any questions, please [contact us](http://csslab.cs.toronto.edu/people/).
