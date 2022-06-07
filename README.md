# Exemplars-guided Empathetic Response Generation Controlled by the Elements of Human Communication

:fire::fire::fire: [Read the paper](https://arxiv.org/pdf/2106.11791.pdf)

# Experiments

Unzip the `data.zip` file.

Train the empathy classifier and sentiment regression models using the following commands:

```
CUDA_VISIBLE_DEVICES=0 python train_empathy_classifier.py --epochs 12 --dim "emo" --lr 1e-5
CUDA_VISIBLE_DEVICES=0 python train_sentiment_regressor.py --epochs 12 --lr 3e-5
```

You can downlaod our empathy and sentiment models from the link given [here](saved/README.md). These pre-trained weights are used for training the main LEMPEx model. The model paths are hardcoded [here](https://github.com/declare-lab/exemplary-empathy/blob/main/models.py#L15) and [here](https://github.com/declare-lab/exemplary-empathy/blob/main/models.py#L18) in `models.py`. 

The main LEMPEx model can be trained using:

```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 12 --lr 1e-5
``` 

An example of retrieving exemplars with a non fine-tuned DPR model is provided in `dpr_exempler_retriever.py`. 

We also experiment with exemplars obtained from a DPR model trained on the Empathetic Dialogues and Empathy Mental Health dataset. We follow the instructions in the [original implementation of DPR](https://github.com/facebookresearch/dpr) for training this model. The main LEMPEx model is trained with exemplars from the trained DPR model

# Overview of the Model

![Alt text](assets/lempex.png?raw=true "Model Architecture")

# Results

Comparing efficacy of our model LEMPEx against the baseline models on various automated and human-evaluated metrics.

![Alt text](assets/results.png?raw=true "Results")

# Case Studies

Comparing responses between models.

![Alt text](assets/compare1.png?raw=true "Compare1")

Comparison of responses with and without DPR exemplars.

![Alt text](assets/compare2.png?raw=true "Compare2")

Comparison of responses with and without empathetic losses.

![Alt text](assets/compare3.png?raw=true "Compare3")

Top exemplars from the DPR model fine-tuned on Empathetic Dialogs, and the original pre-trained DPR checkpoint without any further training. The
exemplars from the fine-tuned DPR model are considerably more empathetic, diverse and contextually relevant. Notably, exemplars from the fine-tuned DPR are
not always semantically similar to the references, although they are stylistically plausible and relevant with respect to the context.

![Alt text](assets/compare4.png?raw=true "Compare4")

# Citation

Majumder, Navonil, Deepanway Ghosal, Devamanyu Hazarika, Alexander Gelbukh, Rada Mihalcea and Soujanya Poria. “Exemplars-guided Empathetic Response Generation Controlled by the Elements of Human Communication.” ArXiv abs/2106.11791 (2021).

