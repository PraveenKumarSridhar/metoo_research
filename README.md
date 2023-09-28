# metoo_research
(Northeastern) Research on social media patterns surrounding MeToo movement

---

### Setup

The code builds its own isolated Conda environments to run each component of the pipeline, and thus requires minimal manual setup. 

**Note**: The `infer_demographics` component requires an older version of tensorflow which is not compatible with Apple M1 chips. Affected users will not be able to run this component. 

#### conda
`conda` is a basic requirement of any system looking to run the code: see [conda's install docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for instructions on how to install for your system, and [Northeastern's RC-DOCS](https://rc-docs.northeastern.edu/en/latest/software/conda.html#working-with-a-miniconda-environment) for info more specifically tailored to the Discovery Cluster. 

#### Requirements
It is recommended to create a new `conda` environment to run the code. The following has instructions for how to run the pipeline in the discovery cluster.
```
cd metoo_research/
sbatch scripts/setup_env.script
```

This will setup the environment for the pipeline.

#### Data
The `data` folder hosts the following:
-  two files used for tagging business entities
-  1 file for tagging news outlets.
-  1 file for tagging celebrities
-  this folder should also be the location for the raw data ZIP file (NOT available from this repository).

---

### Run the code
The project is built using `hydra`, with a central `config.yaml` file that sets key pipeline parameters. Most components will take some kind of input, which is generally the output of the preceding component; other components have parameters that can be set/modified here as well. 

To change the value of any parameter at runtime, use the optional `-P` flag followed by the desired key/value. 

To allow for rapid testing/debugging, the `samp_size` parameter of the consolidate_data component is preconfigured to a value of 1. Remove this parameter from the main `config.yaml` file, or set it to -1, to run the production pipeline. 

**NOTE**: Runtimes will be much longer when running the code from scratch, as it takes some time to configure/install the conda environments of each component. 

#### In Discovery Cluster

It is not recommended to run Discovery Cluster jobs in the same way as the local environment, because any connection interruptions will automatically terminate the session. Batch scripts should be used instead, which run independently of the local session. There are several pre-configured batch scripts in the `scripts` directory: `main.script` will run the entire pipeline, and other scripts run a single component. These scripts should be run using the `sbatch` command (see [RC-DOCS](https://rc-docs.northeastern.edu/en/latest/using-discovery/sbatch.html)); 

To run a single component like `consolidate_data`, use the following command:
```
sbatch scripts/consolidate_data.py
```

To run all components, use the following:
```
sbatch scripts/main.script
```


For debugging purposes, job results will be output in the `runs` directory of the root folder. 

**NOTE**: Batch scripts by default run from the base conda environment, which therefore must be configured with mlflow. To run from a different environment, insert the `conda activate [env_name]` command into the batch script. 

---

### Notebooks
Analysis notebooks are also included in the `notebooks` directory. 
* `lda_vis` is meant as a testing ground for selecting the best LDA model
* `analysis` contains any analysis on the final pipeline artifact (i.e. processed/labeled tweets with topic scores).
* `stm_comp.R` contains an R script to run structured topic modeling on the combined data from `metoo_dataset` and `companies_dataset`

If running these notebooks in the Discovery Cluster, it is recommended to specify the conda PATH when starting the notebook compute cluster.
