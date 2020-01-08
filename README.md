# RSNA Intracranial Hemorrhage Detection

This is the project for [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) hosted on Kaggle in 2019.

<img src="https://i.ibb.co/8m4Bnzr/Screenshot-from-2019-12-07-07-41-47.png" alt="Screenshot-from-2019-12-07-07-41-47" border="0">
<img src="https://i.ibb.co/YWmkWQr/Screenshot-from-2019-12-07-07-40-52.png" alt="Screenshot-from-2019-12-07-07-40-52" border="0">

**Team**
- [Mobassir](https://www.kaggle.com/mobassir) 
- [Mukharbek Organokov](https://www.kaggle.com/muhakabartay)
- [Adity Agur](https://www.kaggle.com/adityaguru149) 
- [Manoj](https://www.kaggle.com/mks2192) 

**Final Solution**
EfficientNet b7. 5-folds. Dice images + preprocessing.
Final position 127th of 1345 teams.
Check our models at [conf](conf/) directory.

**Official baseline/ starter code from @Aappian42**: https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage

<br>

## Requirements

- Python 3.6.6
- [Pytorch](https://pytorch.org/) 1.1.0
- [NVIDIA apex](https://github.com/NVIDIA/apex) 0.1 (for mixed precision training)

## Performance (Single model)

| Backbone | Image size | LB |
----|----|----
| se\_resnext50\_32x4d | 512x512 | 0.070 - 0.072 |
| se\_resnext50\_32x4d | 1024x1025 | 0.070 - 0.071 |
| se\_resnext101\_32x4d | 512x512 | 0.070 |


