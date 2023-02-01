The code for fine-tuning the DPR model is adopted from the [original DPR implementation](https://github.com/facebookresearch/dpr). Note that the original DPR code is distributed with the CC-BY-NC 4.0 license.

Executing the DPR fine-tuning script requires an old version of pytorch (tested on 1.5.0 - 1.7.0) and transformers (tested on 3.0.0). The full requirements are given in the `requirements.txt` file. We suggest that you create a new python environment and install the required packages for fine-tuning the DPR models.

You first need to download the NQ pretrained checkpoint:

```
bash download_checkpoint.sh
```

The data for fine-tuning the DPR models can be prepared using the `prepare_dpr_data.py` script. It will create the training and validation files for Empathetic Dialogues, reddit portion of Empathy Mental Health, and a combination of both the datasets in the `dpr/new_data/`. We have provided the files used in our experiments `dpr/data/` directory.

The training configuration can be found in `conf/train/biencoder_local.yaml` file. You can fine-tune the DPR model on the Empathetic Dialogues as follows:

```
bash train_dpr.sh
```

You can also use the reddit portion of Empathy Mental Health or the combined dataset for DPR fine-tuning. For that, you need to change the `train_datasets` and `dev_datasets` identifier and an appropriate `output_dir` in the `train_dpr.sh` script. The list of dataset identifiers can be found in the dataset configuration file `conf/datasets/encoder_train_default.yaml`.

The fine-tuned model weights would be saved in `outputs/yyyy-mm-dd/aa-bb-cc/saved/` directory. You can then use the saved model in our `../dpr_exempler_retriever.py` script to retrieve the relevant exemplars for the Empathetic Dialogues dataset.