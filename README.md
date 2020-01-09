# Using train_surrogate_models

This is a brief overview in how to use the `train_surrogate_models` module.
The idea behind this module is to generate a set of surrogate models for your particular application.
These models can then be queried to determine the most applicable surrogate model.
There are a few built in functions which will allow you to optimize hyper-parameters for your surrogate models.

Using `train_surrogate_models` can be done in two ways; via a script, as seen in `main.py` or via an interactive terminal such as iPython or Jupyter notebook.
Using a script is advantageous for those who already know what they want their data to look like and will allow users to easily transfer surrogate models to other programs.
Using an interactive terminal will allow users to get a feel for the surrogate models and will create an easy avenue for visualization.

## Getting Started

The first step in using `train_surrogate_models` is importing the module, typically imported as `tm` for train model.
You will then need access to your particular set of data.
The structure for `train_surrogate_models` is identical to `sklearn`, which should make it easy to incorporate if you are already familiar with `sklearn` [1].
Data will typically take the form of `input_variables` (also called design variables, independent variables, etc.) and `output_variables` (also called objective variables, dependent variables, etc.).
`input_variables` will be similar to a list where each entry in the list is a tuple containing the input variables for the problem.
`output_variables` follows a similar format, where each entry in the list is a tuple containing the output variables for the corresponding input variables.
*Note* The list of `input_variables` and `output_variables` must be the same length and the position in the list for each input should be the same as the position for each output.
This format may be updated in the future to just allow for `pandas` dataframes.

Once the data is ready, create an instance of the `Surrogate_Models` class.

`sm = tm.Surrogate_Models()`

You then need to load your data (input/output variables) into the class via the `update_database` method.

`sm.update_database(input_variables, output_variables)`

You can then either create a pre-made model via the `update_model` ,method, or add a new model via `add_model` method.
Currently available pre-made models include (model name is given in ()): linear regression ('lr'), polynomial regression ('pr'), multi-adaptive regression splines ('mars'), gaussian process regression ('gpr'), artificial neural network ('ann'), and random forest regressor ('rf').

`sm.update_model('ann')`

When a model is added it automatically split, scales, and trains the surrogate model on the data set.
The `predict` method can then be used to determine an unknown set of outputs based on a given input.

`sm.predict('ann', [(0,0,0)])`

If you add data to your data set at any point, you can use the `update_database` method in conjunction with the `update_all_models` method to retrain the models with the new data set.
Or if you only want to update one model you can use the `update_model` method.

`sm.update_database(input_variables, output_variables)`
`sm.update_all_models()`
or
`sm.update_model('ann')`

One last thing for the basics.
If you ever need to clear your surrogate model, the method `clear_surrogate_model` will clear all of your input/output variables.
This will leave the model types and hyper-parameters untouched, but remove any associated data.

## Optimizing Surrogate Models

Often times you will find yourself using a surrogate model and needing to tweek the hyper-parameters to create a better model.
`train_surrogate_models` has a few built in functions to help you optimize your surrogate model and select the most qualified surrogate model if multiple are present.

To optimize the hyper-parameters for a surrogate model you can use the `optimize_model` method which will optimize the given model with a set of pre-defined hyper-parameters.

`sm.optimize_model('ann')`

Of you can create a set of hyper-parameters yourself by examining the `sklearn` page and determining what parameters are important.
This is done by adding the `hp` keyword into the `optimize_model` method, where the keyword is followed by a dictionary of `<hyper-parameter>` : `(values)`.

`sm.optimize_model('ann', hp={hidden_layer_sizes: (2,4,10)})`

If you find yourself using multiple surrogate models, you can use the `return_best_model` method to evaluate the R-squared value for each model available.
This will return the name of the best surrogate model available based on the R-squared values

`sm.return_best_model()`

One last aspect of `train_surrogate_models` is the ability to determine what effect hyper-parameters have on your model.
This is done using the `plot_validation_curve` method.

`sm.plot_validation_curve('ann', 'hidden_layer_sizes', [1,2,3,4,5])`

This will test each of these hyper parameters and generate a validation curve to show the effect this parameter has on your model.

## References

[1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
[2] W. McKinney., “‘Data Structures for Statistical Computing in Python”’,Proceedings of the 9th Python in Science Conference, 51-56 (2010).
