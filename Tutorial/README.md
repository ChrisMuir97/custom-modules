# Tutorial 1: Use existing components

A component is self-contained set of code that performs one step in machine learning pipeline, such as data preprocessing, model training, model scoring and so on. A component is analogous to a function, in that it has a name, parameters, expects certain input and returns some value. Any python script can be wrapped as a component following the [component spec](https://github.com/chris-muir/AzureMachineLearningGallery/blob/main/tutorial/component-spec-definition.md).

Azure Machine Learning Gallery contains rich components and pipelines for common machine learning tasks. It can accelerate AI adoption by enabling enterprises and individuals to easily leverage best work of the community instead of starting from scratch.

In this tutorial, you will learn how to build a machine learning pipeline with existing components in the gallery in 2 steps:
 1. Register the component to your Azure Machine Learning workspace.
 2. Build a pipeline using the registered component and built-in modules in Azure Machine Learning designer.

> **! NOTE:**  
>
> **Components** equals to **Modules** in Azure Machine Learning studio UI.

This tutorial will use Wikipedia views forecasting as an example. The related components can be found under the [Prophet](https://github.com/chris-muir/aml-components/tree/main/Prophet) section.
