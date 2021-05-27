# Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection

## Usage
A simple example of running our proposed method is shown as follows.
```python
python FEAWAD.py --network_depth=4 --runs=10 --known_outliers=30 --cont_rate=0.02 --data_format=0 --output=./results.csv
````
See FEAWAD.py for more details about each argument used in this line of code.

The key packages and their versions used in our algorithm implementation are listed as follows
* python==3.6.12
* keras==2.3.1
* tensorflow-gpu==1.13.1
* scikit-learn==0.20.0
* numpy==1.19.4
* pandas==1.1.5
* scipy==1.5.2

See the full paper for the implemenation details of our proposed method.

## Full Paper
The full paper can be found in IEEE Xplore or [arXiv](https://arxiv.org/abs/2105.10500)

## Datasets
The datasets used in our paper are available at the "dataset" folder.

## Citation
> Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu. Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection, IEEE Transactions on Neural Networks and Learning Systems, 2021.

## Contact
If you have any question, please email to Prof. Yingjie Zhou (email: yjzhou09@gmail.com) or Mr. Fanxing Liu (email: finwarrah@gmail.com).
