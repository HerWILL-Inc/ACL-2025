# Official implementation of the ACL 2025 paper "Assessing Gender Bias of Pretrained Bangla Language Models in STEM and SHAPE Fields"

This repository contains code for HerWILL STEM Research team's research paper titled "Assessing Gender Bias of Pretrained Bangla Language Models in STEM and SHAPE Fields" accepted at the GeBNLP workshop under ACL 2025.

![graphical abstract (1)](https://github.com/user-attachments/assets/64914bd3-b182-4a1f-a887-991524cea15e)

You can find our dataset on mendeley data.

https://data.mendeley.com/datasets/y3x569kk9t/2

Paper link will be posted once the camera ready version comes online.

# Usage
It is recommended to open a virtual env.
Install the requirements
```
pip install -r requirements.txt
```
Install normalizer
```
pip install git+https://github.com/csebuetnlp/normalizer
```
run the script
```
python  analyze_bias.py --modelname "csebuetnlp/banglabert"
```

