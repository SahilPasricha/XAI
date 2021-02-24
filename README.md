# XAI
  
  # Visually Explaining the Weight Distribution of Neural Networks Over Time<h1>

This repository contains the source code for [_explAIner_](https://www.researchgate.net/publication/344719862_Visually_Explaining_the_Weight_Distribution_of_Neural_Networks_Over_Time) -- the framework for explainable AI and interactive machine learning.
The aim is to make use of intrinsic features to come up with a domain, data, and model independent method. 

## Feature Space
To eplain the ML models at fundamental levels, there is possibility of exploring various factors in feature space including bias,weights, activation function, inputs or the back prorgation of error.Amongst all these weights hold a direct corelation with model performance, accruacy, inputs as well as outputs. Thus visualizing single feature can have multiple findings.Refer to the reserach article for detailed info.


## Architecture

The framework consists of four plugins, which represent the _stages of explanation_, namely

* Extracting Weights
* Weights Preprocessing
* Non temporal distance analysis 
* Agglomerative clustering
* Diagnosis using Viusialzation
* Refinement
* Hypothesis formaiton
* Reporting
<! --- * Impact of regression on models  --- !>



## Repository Structure

### Folders 
The repository contains 4 folders:

* `backend/`
  This folder contains the python backend for the high-level (model in-/output) explanations.
  
  ```comment
 ML algo tweaked to publish weights with CNN and Auto encoders
```

* `viz/`  
  This folder contains the actual visualization code. It has the following options:  
    * `_1_central_data_tendency/`  
      Plugin for understanding. Data-independent explanations.
    * `_2_binning/`  
      Plugin for diagnosis. Debugging of NN graph.
    * `_3_sorting/`  
      Plugin for refinement. Recommendations on improvements.
    * `_4_dynamic_time_warping_agglomerative_clustering/`  
      Plugin for reporting. Summarizes the findings from previous steps.
    * `_5_location_tendency/`  
      Parts that are used in more than one plugin.
  * `knime/`  
    The modified TensorBoard executable, with explAIner plugins injected.
    


## Extracting weights from model at every training step

Tensorflow supports weight extraction using .get_weights() but this extracts weights of last training step only.
For this analysis, Weights througout the project are required and this can be done by writing a custom call back.


```Bash
docker-compose up --build --remove-orphans explainer_summary
```

To build and start the explAIner TensorBoard executable (together with custom backend servers):

```Bash
docker-compose up --build --remove-orphans -d explainer_tensorboard
```
Although the containers should be up and running after a few seconds, it might take a while until the code is fully compiled and the system gets available under `http://127.0.0.1:6006`.

## Citing this Repository
To reference this repository, please cite the original explAIner publication (pre-print available on [_researchgate.org](https://arxiv.org/abs/1908.00087)):

```
T. Spinner, U. Schlegel, H. Schafer, and M. El-Assady, “explAIner: A Visual Analytics Framework for Interactive and Explainable Machine Learning,” IEEE Trans. on Vis. and Computer Graphics, vol. 26, no. 1, Art. no. 1, 2020, doi: 10.1109/tvcg.2019.2934629.
```

### BibTeX

```
@ARTICLE{SpinnerEtAl2020,
  author = {Thilo Spinner and Udo Schlegel and Hanna Schäfer and Mennatallah El-Assady},
  title = {{explAIner}: A Visual Analytics Framework for Interactive and Explainable Machine Learning},
  journal = {{IEEE} Transactions on Visualization and Computer Graphics},
  year = {2020},
  volume = {26},
  number = {1},
  pages = {1064-1074},
  doi = {10.1109/TVCG.2019.2934629},
}
```
