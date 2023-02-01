# Exemplars-guided Empathetic Response Generation Controlled by the Elements of Human Communication

:fire::fire::fire: [Read the paper](https://arxiv.org/pdf/2106.11791.pdf)

# Experiments

Unzip the `data.zip` file.

Train the T5 and T5-GloVe empathy classifier and sentiment regression models using the following commands:

```
## T5 Models ##
CUDA_VISIBLE_DEVICES=0 python train_empathy_classifier.py --epochs 12 --model "t5" --lr 1e-5 --dim ["emo"|"exp"|"int"]
CUDA_VISIBLE_DEVICES=0 python train_sentiment_regressor.py --epochs 12 --model "t5" --lr 3e-5

## T5-GloVe Models ##
CUDA_VISIBLE_DEVICES=0 python train_empathy_classifier.py --epochs 15 --model "glove-t5" --lr 2e-5 --dim ["emo"|"exp"|"int"]
CUDA_VISIBLE_DEVICES=0 python train_sentiment_regressor.py --epochs 15 --model "glove-t5" --lr 2e-5
```

You can downlaod our empathy and sentiment models from the link given [here](saved/README.md). These pre-trained weights are used for training the main empathetic response generator models. The model paths are hardcoded in `ERGMainModel` in `models.py` and `ERGGloVeMainModel` in `glove_models.py`. 

The main T5 and the T5-GloVe emapthetic response generator models can be trained using:

```
## T5 Empathatic Response Generator Model ##
CUDA_VISIBLE_DEVICES=0 python train_t5.py --epochs 15 --lr 1e-4

## T5-GloVe Empathatic Response Generator Model ##
CUDA_VISIBLE_DEVICES=0 python train_glove_t5.py --epochs 50 --lr 1e-4 --add-exemplars "glove-t5"
``` 

The code for fine-tuning the DPR model is provided in the `DPR/` directory. You can follow the instructions in `DPR/` directory to fine-tune a DPR model on the Empathetic Dialogues and/or Empathy Mental Health Dataset. Then you can use the fine-tuned model path to retrieve the exemplars using:

```
CUDA_VISIBLE_DEVICES=0 python dpr_exempler_retriever.py --path DPR/outputs/yyyy-mm-dd/aa-bb-cc/saved/empd/dpr_biencoder.0
```

If you do not pass a `--path` then the non fine-tuned DPR model will be used for retrieval. We have provided the fine-tuned DPR retrieved examples in the `*_dpr.csv` files in the `data/empathetic_dialogues/` directory.

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

Majumder, Navonil, Deepanway Ghosal, Devamanyu Hazarika, Alexander Gelbukh, Rada Mihalcea and Soujanya Poria. “Exemplars-guided Empathetic Response Generation Controlled by the Elements of Human Communication.” IEEE Access (2022).

