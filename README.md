# EGFR_PIC50
Classifier for active compounds binding to the Epidermal Growth Factor Receptor (EGFR).
As the task is to distinguish compounds that have pIC50>8, a classifier approach
is the most natural.

The jupyter notebook files in this repository track my desicion-making process
for building a classifier to distinguish active (pIC50>8) from inactive molecules
for binding with EGFR. While the prediction pipeline is in a standalone file.

- prediction_pipeline.py runs the prediction pipeline by reading SMILES srings from
an input file. It has built in documentation acessible via "predictionPipeline.py -h".
It expects to be run in the curent folder, as it loads saved models and feature
engineering information from local files. It can be tested with "predTest.smi",
its default input and can output SMILES of identified actives to screen or file.

- Step1_DatasetExploration.ipynb looks at the relevant experimental dataset and
builds RDKit molecule instances for all SMILES strings in it. This file also
generates 3D conformations for them.

- Step2_GenerateFeatures.ipynb generates both 2D and 3D descriptors for the molecules
with the help of a pyTorch Dataset subclass defined in custom_mol_dataset.py.

- custom_mol_dataset.py defines a pyTorch Dataset subclass that precomputes most molecular
descriptors available in RDKit and chaches them in an hdf5 file for quick retriaval.
It also supports feature selection via an index filter, feature normalization,
as well as an in-memory cache for the normalized data, when it fits into RAM.
This file is adapted from one of my previous projects (https://github.com/zetadin/TransferLearningFromPLS).

- Step3_TrainSimpleModels.ipynb goes through a number of classifier model types available in
scikit-learn and compares their performance on feature sets of both 2D and 2D+3D descriptors.
Including 3D descriptors reduces performance of most models. Best performance is obtained with
a Support Vector Machine classifier with a Radial Basis Function kernel.
Variations of it are also tested with primary component analysis (PCA) and
partial least squares (PLS) driven dimentinality reduction of the feature space.
Optimal results are obtained with a PLS transformation of the features,
which drastically reduces execution time while marginally improving the F1 score.
Therefore, this approach is selected for the prediction pipeline.

In the model selection procedure, performance is quantified primarily via the F1 score,
which offers a good tradeoff between recall and precision needed in an effective binary classifier,
especially if one is only interested in retrieving one of the classes.
Separate precision, recall and AUC scores are also recorded for confirmation,
though they were not used for ranking.
Hyper-parameter searches were performed using 5-fold cross-validation for models with
hyper-parameters. Training samples were weighted to compensate class imballance where model
implementations support this.

My previous experience had shown that including 3D descriptors derrived from docked molecular
structures in adition to coordinate independent descriptors
improved accuracy of neural network regression models of binding affinity.
Therefore, I took this project as an opportunity to test if they are still helpful without docking.
The unsurprising conclusion is no, they are not.


Please use prerequisite module versions specified in requirements.txt: 
	pip3 install -r requirements.txt
Newer versions of RDKit, for example, raise a warning about Crippen logP calculations
when building descriptors.
