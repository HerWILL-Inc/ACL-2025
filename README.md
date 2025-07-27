# Official implementation of the ACL 2025 paper "Assessing Gender Bias of Pretrained Bangla Language Models in STEM and SHAPE Fields"

This repository contains code for HerWILL STEM Research team's research paper titled "Assessing Gender Bias of Pretrained Bangla Language Models in STEM and SHAPE Fields" accepted at the GeBNLP workshop under ACL 2025 (A* conference).

![graphical abstract (1)](https://github.com/user-attachments/assets/64914bd3-b182-4a1f-a887-991524cea15e)

You can find our dataset on mendeley data.

https://data.mendeley.com/datasets/y3x569kk9t/2

Give our paper a read at
https://aclanthology.org/2025.gebnlp-1.24/

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

# Citation
If you use our code/ ideas from our paper, please cite
```
@inproceedings{arnob-etal-2025-assessing,
    title = "Assessing Gender Bias of Pretrained {B}angla Language Models in {STEM} and {SHAPE} Fields",
    author = "Arnob, Noor Mairukh Khan  and
      Mahmud, Saiyara  and
      Wasi, Azmine Toushik",
    editor = "Fale{\'n}ska, Agnieszka  and
      Basta, Christine  and
      Costa-juss{\`a}, Marta  and
      Sta{\'n}czak, Karolina  and
      Nozza, Debora",
    booktitle = "Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.gebnlp-1.24/",
    pages = "268--281",
    ISBN = "979-8-89176-277-0",
```


