## Deputies Watcher: French deputies votes prediction

Authors: *Florian Eisenbarth, Nicolas Oulianov, Nicolas Boussenina, Armand Foucault, Eloi Alardet*

This class group project is about predicting positions of deputies on votes in the French National Assembly.

### Description

We collect data from the government and heavily process it to build a custom supervized dataset. 

We analyze this dataset to highlight *political games* inside the National Assembly.

Finally, using a Deep Learning model and smart feature engineering, we predict voting positions of a set of 10 political groups. We reach an accuracy of roughly 75%. We create a custom metric, a weighted F1 score tuned to our problem, that scores about 72% (best being 100%).

### Statistical analysis

To get a better understanding of the data and the estimator, reading the [starting kit notebook](starting_kit.ipynb) is a good start. 

### Testing the estimator with RAMP

This repository is built to be a RAMP challenge. To learn about the RAMP library, please visit [the documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html). 

The submissions need to be located in the submissions folder. For instance for my_submission, it should be located in submissions/my_submission.

To run a specific submission, you can use the ramp-test command line:

`ramp-test --submission my_submission`

You can get more information regarding this command line:

`ramp-test --help`

### License

This project is published under a [MIT License](https://opensource.org/licenses/MIT).

