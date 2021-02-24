# eXplainable Artificial Intelligence: Visually Explaining the Weight Distribution of Neural Networks Over Time<h1>

This repository contains the source code for [_XAI_weights_disttribution_](https://www.researchgate.net/publication/344719862_Visually_Explaining_the_Weight_Distribution_of_Neural_Networks_Over_Time) -- the framework for explainable AI and interactive machine learning.
The aim is to make use of intrinsic features to come up with a domain, data, and model independent method. 

## Feature Space
To explain the ML models at fundamental levels, there is possibility of exploring various factors in feature space including bias,weights, activation function, inputs or the back prorgation of error.Amongst all these weights hold a direct corelation with model performance, accruacy, inputs as well as outputs. Thus visualizing single feature can have multiple findings.Refer to the reserach article for detailed info.


## Architecture

The framework consists of 8 steps, which represent the _stages of explanation_, namely
<p align = "center">
  <a href = "#ExtractingWeights">Extracting Weights</a><br>
  <a href = "#WeightsPreprocessing">Weights Preprocessing</a><br>
  <a href = "#DTW">Non temporal distance analysis </a><br>
  <a href = "#DTW">agglomerative clustering clustering</a><br>
  <a href = "#viz">Diagnosis using Visualization</a><br>
  <a href = "#Refine">Refinement</a><br>
  <a href = "#Hypothesis">Hypothesis Testing</a><br>
  <a href = "#Reporting">Reporting</a><br>
</p>

``` Add to it later Impact of regression on models```



## Repository Structure

The repository contains 4 folders:

* `backend/`  <a name="ExtractingWeights"></a>
  This folder contains the python backend for the tensorflow core (model in-/output) alternatiions.
      * `_1_callbacks/`  
        These core Tensorflow files and need manual integration with relevant TF version.
        The purpose is to tweak the save_weights callback enabling weights extraction on every training step.
         

* `viz/`  <a name="viz"></a>
  This folder contains the actual visualization code. It has the following options:  
    * `_1_central_data_tendency/`   <a name="viz"></a>
      Plugin to sort, cluster and visualize model wights over training steps.
      It supports features such as brush to zoom and offer variety of options to explore various aspects of weight data.
    * `_2_binning/`   <a name="WeightsPreprocessing"></a>
       Clusters the weights in number of bins (selected by scroll bar) on the run.
       Single click on given line further presents exploded view of weights inside bin till one line per weight
    * `_3_sorting/`   <a name="viz"></a>
      As weights are dynamic and vary in development, there is no single sorting method that holds for all models. In viz their are options to sort weights.
    * `_4_dynamic_time_warping_agglomerative_clustering/`   <a name="viz"></a>
      Performs DTW on weights to find records that have same pattern of development irrespective of training steps. Then, they are clustered using aglomerative algorithm 

* `knime/`  <a name="viz"></a>
    The KNIME tool pipeline to perform data cleansing and visualize the CDT of model weights.
* `algorithms/`   <a name="DTW"></a>   
    * `_1_dynamic_time_warping/`  
      Plugin for calculating distance between weights in non temporal fashion 
    * `_2_hierarchical_clustering/`  
      Plugin for understanding. Data-independent explanations.

## Extracting weights from model at every training step

Tensorflow supports weight extraction using get_weights() but this extracts weights of last training step only.
For this analysis, Weights througout the project are required and this can be done by writing a custom call back.


```Bash
docker-compose up --build --remove-orphans explainer_summary
```

* `Refinement/`  <a name="Refine"></a>
    The KNIME tool pipeline to perform data cleansing and visualize the CDT of model weights.

* `Hypothesis Testing/`  <a name="Hypothesis"></a>
    The KNIME tool pipeline to perform data cleansing and visualize the CDT of model weights.

* `Reporting/` <a name="Reporting"></a>
    The KNIME tool pipeline to perform data cleansing and visualize the CDT of model weights.


To build and start the explAIner TensorBoard executable (together with custom backend servers):

```Bash
docker-compose up --build --remove-orphans -d explainer_tensorboard
```
Although the containers should be up and running after a few seconds, it might take a while until the code is fully compiled and the system gets available under `http://127.0.0.1:6006`.

## Citing this Repository
To reference this repository, please cite the original explAIner publication (pre-print available on [_researchgate.org](https://www.researchgate.net/publication/344719862_Visually_Explaining_the_Weight_Distribution_of_Neural_Networks_Over_Time)):



