# Image Colorization with GANs

The weights of the models can be found in this [Google Drive](https://drive.google.com/drive/folders/1YVphPHLabR7TWE0o9wqoc0BmM7OPSfSN?usp=share_link). You can download them and run inference of any of the trained models or continue training with your custom configurations.

In the `training_results/` directory, you can find the training configurations, training logs, and generated image after each epoch for all trained models. 

## Further training
If you would like to further train the models (with my weights) you can run the following script
```
conda create --name new_env python
conda activate new_env
pip install -r requirements.txt
python3 train.py --output_format "your desired output format" --resnet_backbone "True or False"
```

## Inference
In order to run inference of the trained models, please place all grayscale images into a single directory and model weights into the following directories:
HERE GOES THE TREE
and then run
```
python3 inference.py --images_path "your directory with grayscale images" --output_format "your desired output format" --resnet_backbone "True or False"
```
