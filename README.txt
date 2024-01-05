To add:

Target_transforms for the labels to normalize the labels in each class so that the probabilities all add up to 1.
These Target transforms are to be added in Dataset_initialize.py

Make appropriate changes in Predictions.py so that the normalizations made in the Target_transforms are inverted back so that the labels
for each class do not add up to 1 (i.e. they add up to the Whatever the probability was of the Category that lead to them in the preceding Class)

