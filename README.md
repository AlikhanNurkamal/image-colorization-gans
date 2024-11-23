# Image Colorization with GANs

This is my final project for the course ELCE455 - Machine Learning with Python.

In the `training_notebooks/` directory you can find 4 Jupyter notebooks that were used to train 4 GAN models. I ran these notebooks in Kaggle since it provides free GPU.

In the `training_results/` directory you can find the training configurations, training logs, and generated image after each epoch for all trained models.

## Model weights
The weights of the models can be found in this [Google Drive](https://drive.google.com/drive/folders/1YVphPHLabR7TWE0o9wqoc0BmM7OPSfSN?usp=share_link). You can download them and run inference of any of the trained models or continue training with your custom configurations. In Google Drive, the directory `default/` contains model weights of both discriminator and generator **without** the pre-trained ResNet-18 backbone, whereas the directory `resnet/` stores weights of both models with the ResNet-18 backbone.

## Further training
If you would like to further train the models (with my weights) you should:
1. Change the `config.py` file by setting the `load_models` attribute to True and the `start_epoch` attribute to 50. If you wish, you can modify other configurations as well.
2. Specify the directory to Discriminator and Generator weights (that you downloaded from the Google Drive) in lines 207 and 208 of the `train.py` script.
3. Run the following script
```
conda create --name new_env python
conda activate new_env
pip install -r requirements.txt
python3 train.py --output_format "your desired output format" --resnet_backbone "True or False"
```

## Inference
In order to run inference of the trained models, please place all grayscale images into a single directory and model weights into the `checkpoints/` directory, which should follow the following structure:
```
.
├── ...
├── inference.py
├── checkpoints
│   ├── default
│   │   ├── ab
│   │   │   ├── disc_checkpoints_best.pth.tar
│   │   │   └── gen_checkpoints_best.pth.tar
│   │   └── rgb
│   │       ├── disc_checkpoints_best.pth.tar
│   │       └── gen_checkpoints_best.pth.tar
│   └── resnet
│       ├── ab
│       │   ├── disc_checkpoints_best.pth.tar
│       │   ├── gen_checkpoints_best.pth.tar
│       └── rgb
│           ├── disc_checkpoints_best.pth.tar
│           ├── gen_checkpoints_best.pth.tar
```
and then run
```
python3 inference.py --images_path "your directory with grayscale images" --output_format "your desired output format" --resnet_backbone "True or False"
```
## Streamlit app
In case you would like to run your local server and colorize images in a web application, run in the terminal
```
streamlit run app.py
```
Note that in order for the web app to work, the model checkpoints should be stored in the same structure explained above!
