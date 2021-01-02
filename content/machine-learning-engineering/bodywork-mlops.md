Title: Deploying Python ML Models with Bodywork
Date: 2020-12-01
Tags: python, machine-learning, mlops, kubernetes, bodywork

![bodywork_logo]({static}/images/machine-learning-engineering/bodywork/bodywork-logo.png)

Solutions to Machine Learning (ML) tasks are often developed within Jupyter notebooks. Once a candidate solution is found, you are then faced with an altogether different problem - how to engineer the solution into your product and how to maintain the performance of the solution as new instances of data are experienced.

## What is this Tutorial Going to Teach Me?

* How to start with a solution to a ML task developed within a Jupyter notebook, and map it into two separate Python modules: one for training a model and one for deploying the trained model as a model-scoring service.
* How to execute these `train` and `deploy` modules (that form a simple ML pipeline), remotely on a [Kubernetes](https://kubernetes.io/) cluster, using [GitHub](https://github.com/) and [Bodywork](https://bodywork.readthedocs.io/en/latest/).
* How to interact-with and test the model-scoring service that has been deployed to Kubernetes.
* How to run the pipeline on a schedule, so that the model is periodically re-trained and then re-deployed (when new data is available), without the manual intervention of a machine learning engineer.

**Table of Contents**

[TOC]

## Introduction

I’ve written at length on the subject of getting machine learning into production - an area now referred to as Machine Learning Operations (MLOps). MLOps is a hot topic within the field of machine learning engineering. For example, my [blog post]({filename}k8s-ml-ops.md) on *Deploying Python ML Models with Flask, Docker and Kubernetes* is viewed by hundreds of machine learning practitioners every month; at the recent [Data and AI Summit](https://databricks.com/dataaisummit/europe-2020/agenda?_sessions_focus_tax=productionizing-machine-learning) there was an entire track devoted to ‘Productionizing Machine Learning’; Thoughtwork’s [essay](https://www.thoughtworks.com/insights/articles/intelligent-enterprise-series-cd4ml) on *Continuous Delivery for ML* is now an essential reference for all machine learning engineers, together with Google’s [paper](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) on the *Hidden Technical Debt in Machine Learning Systems*; and MLOps even has its own entry on [Wikipedia](https://en.wikipedia.org/wiki/MLOps).

### Why is MLOps Getting so Much Attention?

In my opinion, this is because we are at a point where a significant number of organisations have now overcome their data ingestion and engineering problems. They are able to provide their data scientists with the data required to solve business problems using machine learning, only to find that, as Thoughtworks put it,

> “*Getting machine learning applications into production is hard*”

To tackle some of the core complexities of MLOps, many machine learning engineering teams have settled on approaches that are based-upon deploying containerised models, usually as RESTful model-scoring services, to some type of cloud platform. Kubernetes is especially useful for this as I have [written about before]({filename}k8s-ml-ops.md).

### MLOps with Bodywork

Running machine learning code in containers has become a common pattern to guarantee reproducibility between what has been developed and what is deployed in production environments.

Most machine learning engineers do not, however, have the time to develop the skills and expertise required to deliver and deploy containerised machine learning systems into production environments. This requires an understanding of how to build container images, how to push build artefacts to image repositories and how to configure a container orchestration platform to use these to execute batch jobs and deploy services.

Developing and maintaining these deployment pipelines is time-consuming. If there are multiple projects, each requiring re-training and re-deployment, then the management of these pipelines will quickly become a large burden.

This is where the Bodywork framework steps-in - to take responsibility for pulling you machine learning projects into containers and deploying them to the Kubernetes container orchestration platform. Bodywork can ensure that your projects are always trained with the latest data, the most recent models are always deployed and your machine learning systems remain generally available.

![bodywork_logo]({static}/images/machine-learning-engineering/bodywork/ml-pipeline.png)

Bodywork is a tool built upon the Kubernetes container orchestration platform and is aimed at machine learning engineers to help them:

* **continuously deliver** - code for preparing features, training models, scoring data and defining model-scoring services. Bodywork containers running on Kubernetes will pull code directly from your project's Git repository, removing the need to build-and-push your own container images.
* **continuously deploy** - batch jobs, model-scoring services and complex machine learning pipelines, using the Bodywork workflow-controller to orchestrate end-to-end machine learning workflows on Kubernetes.

In other words, Bodywork automates the repetitive tasks that most machine learning engineers think of as [DevOps](https://en.wikipedia.org/wiki/DevOps), allowing them to focus their time on what they do best - machine learning.

This post serves as a short tutorial on how to use Bodywork to productionise a common ML pipeline - train-and-deploy. This tutorial refers to files within a Bodywork project hosted on GitHub - see [bodywork-ml-pipeline-project](https://github.com/bodywork-ml/bodywork-ml-pipeline-project).

### Prerequisites

If you want to execute the example code, then you will need:

* to [install](https://bodywork.readthedocs.io/en/latest/installation/) the bodywork Python package on your local machine.
* access to a Kubernetes cluster - either a single-node on your local machine using [minikube](https://minikube.sigs.k8s.io/docs/) or [Docker-for-desktop](https://www.docker.com/products/docker-desktop), or as a managed service from a cloud provider, such as [EKS](https://aws.amazon.com/eks) from AWS or [AKS](https://azure.microsoft.com/en-us/services/kubernetes-service/) from Azure.
* [Git](https://git-scm.com) and a basic understanding of how to use it.

Familiarity with basic [Kubernetes concepts](https://kubernetes.io/docs/concepts/) and some exposure to the [kubectl](https://kubernetes.io/docs/reference/kubectl/overview/) command-line tool will make life easier. The introductory article I wrote on [*Deploying Python ML Models with Flask, Docker and Kubernetes*]({filename}k8s-ml-ops.md), is a good place to start.

## A Machine Learning Task

The ML problem we have chosen to use for this tutorial, is the classification of iris plants into one of their three sub-species, given their physical dimensions. It uses the [iris plants dataset](https://scikit-learn.org/stable/datasets/index.html#iris-dataset) and is an example of a multi-class classification task.

The Jupyter notebook titled [ml_prototype_work.ipynb](https://github.com/bodywork-ml/bodywork-ml-pipeline-project/blob/master/ml_prototype_work.ipynb) and found in the root of the [bodywork-ml-pipeline-project](https://github.com/bodywork-ml/bodywork-ml-pipeline-project) repository, documents the trivial ML workflow used to arrive at a proposed solution to this task, by training a Decision Tree classifier and persisting the trained model to cloud storage. Take five minutes to read through it.

## A Machine Learning Operations Task

![train_and_deploy]({static}/images/machine-learning-engineering/bodywork/concepts_train_and_deploy.png)

Now that we have developed a solution to our chosen ML task, how do we get it into production - i.e. how can we split the Jupyter notebook into a 'train-model' stage that persists a trained model to cloud storage, and a separate 'deploy-scoring-service' stage that will load the persisted model and start a web service to expose a model-scoring API?

The solution using Bodywork is packaged as a [GitHub repository](https://github.com/bodywork-ml/bodywork-ml-pipeline-project), whose root directory is as follows,

![example_project_root]({static}/images/machine-learning-engineering/bodywork/example-project-root.png)

Bodywork ML projects must be packaged as Git repositories, using the structure described in this tutorial, from where pre-built Bodywork containers running on Kubernetes (k8s), can pull them for deployment. There are no build artefacts - such as Docker images - that need to be built as part of the deployment process.

The sub-directories contain all the code required to run a single stage - for example, in the `stage-1-train-model` directory you will find the following files,

![train_model_stage]({static}/images/machine-learning-engineering/bodywork/train-model-stage.png)

And similarly, in the `stage-2-deploy-scoring-service` directory you will find the following files,

![deploy_model_stage]({static}/images/machine-learning-engineering/bodywork/deploy-model-stage.png)

The remainder of this tutorial is concerned with explaining what is contained within these directories and their files, and how to command Bodywork to work with them to operationalise the solution on Kubernetes.

### Configuring a Bodywork Batch Stage for Training a Model

The `stage-1-train-model` directory contains the code and configuration required to train the model within a pre-built container on a k8s cluster, as a batch workload. Using the [ml_prototype_work.ipynb](https://github.com/bodywork-ml/bodywork-ml-pipeline-project/blob/master/ml_prototype_work.ipynb) notebook as a reference, the `train_model.py` module contains the code required to:

* download data from an AWS S3 bucket;
* pre-process the data (e.g. extract labels for supervised learning);
* train the model and compute performance metrics; and,
* persist the model to the same AWS S3 bucket that contains the original data.

In essence, it can be summarised as,

```python
from datetime import datetime
from urllib.request import urlopen

# other imports
# ...

DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com'
            '/data/iris_classification_data.csv')

# other constants
# ...


def main() -> None:
    """Main script to be executed."""
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    trained_model = train_model(features, labels)
    persist_model(trained_model)


# other functions definitions used in main()
# ...


if __name__ == '__main__':
    main()
```

We recommend that you spend five minutes familiarising yourself with the full contents of [train_model.py](https://github.com/bodywork-ml/bodywork-ml-pipeline-project/blob/master/stage-1-train-model/train_model.py). When Bodywork runs the stage, it will do so in exactly the same way as if you were to run,

```shell
$ python train_model.py
```

And so everything defined in `main()` will be executed.

The `requirements.txt` file lists the 3rd party Python packages that will be Pip-installed on the pre-built Bodywork host container, as required to run the `train_model.py` script. In this example we have,

```text
boto3==1.16.15
joblib==0.17.0
pandas==1.1.4
scikit-learn==0.23.2
```

* `boto3` - for interacting with AWS;
* `joblib` - for persisting models;
* `pandas` - for manipulating the raw data; and,
* `scikit-learn` - for training the model.

Finally, the `config.ini` file allows us to configure the key parameters for the stage,

```ini
[default]
STAGE_TYPE="batch"
EXECUTABLE_SCRIPT="train_model.py"
CPU_REQUEST=0.5
MEMORY_REQUEST_MB=100

[batch]
MAX_COMPLETION_TIME_SECONDS=30
RETRIES=2
```

From which it is clear to see that we have specified that this stage is a batch stage (as opposed to a service-deployment), that `train_model.py` should be the script that is run, together with an estimate of the CPU and memory resources to request from the k8s cluster, how long to wait and how many times to retry, etc.

### Configuring a Bodywork Service Stage for Deploying a Model-Scoring Service

The `stage-2-deploy-scoring-service` directory contains the code and configuration required to load the model trained in `stage-1-train-model` and use it within the definition of a REST API endpoint, that will accept a single instance (or row) of data encoded as JSON in the request, and return the model's prediction as JSON data in the corresponding response.

We have decided to choose the Python [Flask](https://flask.palletsprojects.com/en/1.1.x/) framework with which to create our REST API server. The use of Flask is **not** a requirement in any way and you are free to use different frameworks - e.g. [FastAPI](https://fastapi.tiangolo.com).

Within this stage's directory, `serve_model.py` defines the REST API server containing our ML scoring endpoint. It can be summarised as,

```python
from urllib.request import urlopen
from typing import Dict

# other imports
# ...

MODEL_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com/models'
             '/iris_tree_classifier.joblib')

# other constants
# ...


@app.route('/iris/v1/score', methods=['POST'])
def score() -> Response:
    """Iris species classification API endpoint"""
    request_data = request.json
    X = make_features_from_request_data(request_data)
    model_output = model_predictions(X)
    response_data = jsonify({**model_output, 'model_info': str(model)})
    return make_response(response_data)


# other functions definitions used in score() and below
# ...


if __name__ == '__main__':
    model = get_model(MODEL_URL)
    print(f'loaded model={model}')
    print(f'starting API server')
    app.run(host='0.0.0.0', port=5000)
```

We recommend that you spend five minutes familiarising yourself with the full contents of [serve_model.py](https://github.com/bodywork-ml/bodywork-ml-pipeline-project/blob/master/stage-2-deploy-scoring-service/serve_model.py). When Bodywork runs the stage, it will start the server defined by `app` and expose the `/iris/v1/score` route that is being handled by `score()` (note that this process has no scheduled end).

The `requirements.txt` file lists the 3rd party Python packages that will be Pip-installed on the Bodywork host container, as required to run `serve_model.py`. In this example we have,

```text
Flask==1.1.2
joblib==0.17.0
numpy==1.19.4
scikit-learn==0.23.2
```

* `Flask` - the framework upon which the REST API server is built;
* `joblib` - for loading the persisted model;
* `numpy` & `scikit-learn` - for working with the ML model.

The `config.ini` file for this stage is,

```ini
[default]
STAGE_TYPE="service"
EXECUTABLE_SCRIPT="serve_model.py"
CPU_REQUEST=0.25
MEMORY_REQUEST_MB=100

[service]
MAX_STARTUP_TIME_SECONDS=30
REPLICAS=2
PORT=5000
```

From which it is clear to see that we have specified that this stage is a service-deployment stage (as opposed to a batch stage), that `serve_model.py` should be the script that is run, together with an estimate of the CPU and memory resources to request from the k8s cluster, how long to wait for the service to start-up and be 'ready', which port to expose and how many instances (or replicas) of the server should be created to stand-behind the cluster-service.

### Configuring the Complete Bodywork Workflow

The `bodywork.ini` file in the root of this repository contains the configuration for the whole workflow - a workflow being a collection of stages, run in a specific order, that can be represented by a Directed Acyclic Graph (or DAG).

```ini
[default]
PROJECT_NAME="bodywork-ml-pipeline-project"
DOCKER_IMAGE="bodyworkml/bodywork-core:latest"

[workflow]
DAG="stage-1-train-model >> stage-2-deploy-scoring-service"

[logging]
LOG_LEVEL="INFO"
```

The most important element is the specification of the workflow DAG, which in this instance is simple and will instruct the Bodywork workflow-controller to train the model and then (if successful) deploy the scoring service.

### Testing the Workflow

Firstly, make sure that the [bodywork](https://pypi.org/project/bodywork/) package has been Pip-installed into a local Python environment that is active. Then, make sure that there is a namespace setup for use by bodywork projects - e.g. `ml-pipeline` - by running the following at the command line,

```shell
$ bodywork setup-namespace ml-pipeline
```

Which should result in the following output,

```text
creating namespace=ml-pipeline
creating service-account=bodywork-workflow-controller in namespace=ml-pipeline
creating cluster-role-binding=bodywork-workflow-controller--ml-pipeline
creating service-account=bodywork-jobs-and-deployments in namespace=ml-pipeline
```

Then, the workflow can be tested by running the workflow-controller locally (to orchestrate remote containers on k8s), using,

```shell
$ bodywork workflow \
    --namespace=ml-pipeline \
    https://github.com/bodywork-ml/bodywork-ml-pipeline-project \
    master
```

Which will run the workflow defined in the `master` branch of the project's remote GitHub repository, all within the `ml-pipeline` namespace. The logs from the workflow-controller and the containers nested within each constituent stage, will be streamed to the command-line to inform you on the precise state of the workflow. For example,

```text
2020-11-24 20:04:12,648 - INFO - workflow.run_workflow - attempting to run workflow for project=https://github.com/bodywork-ml/bodywork-ml-pipeline-project on branch=master in kubernetes namespace=ml-pipeline
git version 2.24.3 (Apple Git-128)
Cloning into 'bodywork_project'...
remote: Enumerating objects: 92, done.
remote: Counting objects: 100% (92/92), done.
remote: Compressing objects: 100% (64/64), done.
remote: Total 92 (delta 49), reused 70 (delta 27), pack-reused 0
Receiving objects: 100% (92/92), 20.51 KiB | 1.58 MiB/s, done.
Resolving deltas: 100% (49/49), done.
2020-11-24 20:04:15,579 - INFO - workflow.run_workflow - attempting to execute DAG step=['stage-1-train-model']
2020-11-24 20:04:15,580 - INFO - workflow.run_workflow - creating job=bodywork-ml-pipeline-project--stage-1-train-model in namespace=ml-pipeline
...
```

After a stage completes, you will notice that the logs from within the container are streamed into the workflow-controller logs. For example,

```text
----------------------------------------------------------------------------------------------------
---- pod logs for bodywork-ml-pipeline-project--stage-1-train-model
----------------------------------------------------------------------------------------------------
2020-11-24 20:04:18,917 - INFO - stage.run_stage - attempting to run stage=prepare-data from master branch of repo at https://github.com/bodywork-ml/bodywork-ml-pipeline-project
git version 2.20.1
Cloning into 'bodywork_project'...
Collecting boto3==1.16.15
  Downloading boto3-1.16.15-py2.py3-none-any.whl (129 kB)
...
```

The aim of this log structure, is to provide a reliable way of debugging workflows out-of-the-box, without forcing you to integrate a complete logging solution. This is not a replacement for a complete logging solution - e.g. one based on Elasticsearch - it is intended as a temporary solution to get your ML projects operational quickly.

Note that you can also keep track of the current state of all k8s resources created by the workflow-controller in the `ml-pipeline` namespace, by using the kubectl CLI tool - e.g.,

```shell
$ kubectl -n ml-pipeline get all
```

#### Testing the Model-Scoring Service

Once the workflow has completed, the ML scoring service deployed within your cluster will be ready for testing. Service deployments are accessible via HTTP from within the cluster - they are not exposed to the public internet. To test the service from your local machine you will first of all need to start a proxy server to enable access to your cluster. This can be achieved by issuing the following command,

```shell
$ kubectl proxy
```

Then in a new shell, you can use the curl tool to test the service. For example,

```shell
$ curl http://localhost:8001/api/v1/namespaces/ml-pipeline/services/bodywork-ml-pipeline-project--stage-2-deploy-scoring-service/proxy/iris/v1/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

If successful, you should get the following response,

```json
{
    "species_prediction":"setosa",
    "probabilities":"setosa=1.0|versicolor=0.0|virginica=0.0",
    "model_info": "DecisionTreeClassifier(class_weight='balanced', random_state=42)"
}
```

### Executing the Workflow Remotely on a Schedule

If you're happy with the test results, then you can schedule the workflow-controller to operate remotely on the cluster as a k8s cronjob. To setup the the workflow to run every hour, for example, use the following command,

```shell
$ bodywork cronjob create \
    --namespace=ml-pipeline \
    --name=ml-pipeline \
    --schedule="0 * * * *" \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-ml-pipeline-project \
    --git-repo-branch=master
```

Each scheduled workflow will attempt to re-run the workflow, end-to-end, as defined by the state of this repository's `master` branch at the time of execution - performing rolling-updates to service-deployments and automatic roll-backs in the event of failure.

To get the execution history for all `ml-pipeline` jobs use,

```shell
$ bodywork cronjob history \
    --namespace=ml-pipeline \
    --name=ml-pipeline
```

Which should return output along the lines of,

```text
JOB_NAME                                START_TIME                    COMPLETION_TIME               ACTIVE      SUCCEEDED       FAILED
ml-pipeline-1605214260          2020-11-12 20:51:04+00:00     2020-11-12 20:52:34+00:00     0           1               0
```

Then to stream the logs from any given cronjob run (e.g. to debug and/or monitor for errors), use,

```shell
$ bodywork cronjob logs \
    --namespace=ml-pipeline \
    --name=ml-pipeline-1605214260
```

### Cleaning Up

To clean-up the deployment in its entirety, delete the namespace using kubectl - e.g. by running,

```shell
$ kubectl delete ns ml-pipeline
```

## Where to go from Here

Read the official Bodywork [documentation](https://bodywork.readthedocs.io/en/latest/) or ask a question on the Bodywork [discussion forum](https://bodywork.flarum.cloud/).

## Disclosure

I am one of the co-founders of [Bodywork Machine Learning](https://www.bodyworkml.com)!
