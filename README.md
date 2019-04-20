# CanAI-Name2Vec

## Dependencies
This code requires Python 3.6 or higher, and relies on the following packages:
+ gensim
+ matplotlib
+ numpy
+ scipy

## Generating a Model
To create a model, run:
```
python main.py [epochs] [vector_size] [window]
```
This will generate a model using Doc2Vec and create a histogram to visualize the model performance. The model will be saved to `models/epochs_[epochs]_vectorSize_[vector_size]_window_[window].model` and the histogram will be saved to `histograms/epochs_[epochs]_vectorSize_[vector_size]_window_[window].png`.

The name pairings used to create histograms are randomly generated the first time they are required, at which point they are saved to `cache/index_pairs.json`. These saved name pairings will be reused when generating histograms for additional models.

The distance of each name pair used to create a histogram is also saved. These values can be found in the `matching_name_distance` and `random_name_distance` directories.

## Name Pairs
The data that allowed for this experimentation was made availible by:

[Jeffrey Sukharev, Leonid Zhukov, Alexandrin Popescul "Parallel corpus approach for name matching in record linkage" Proceedings of IEEE ICDM 2014, Shenzhen, China.](https://github.com/jeffsicdm14/name_pairs)
