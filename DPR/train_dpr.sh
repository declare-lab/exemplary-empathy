CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py train_datasets=[empd_train] dev_datasets=[empd_valid] train=biencoder_local output_dir=saved/empd/ --model_file=downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp