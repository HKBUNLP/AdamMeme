# AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Meme Harmfulness

The repository for **AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Meme Harmfulness**.

## setup

Please refer to [LLaVA](https://github.com/haotian-liu/LLaVA)

Install other requirements by：

```bash
pip install -r requirements.txt
```

## data
For data used in our paper, please refer to [MAMI](https://github.com/TIBHannover/multimodal-misogyny-detection-mami-2022), [HarM](https://github.com/LCS2-IIITD/MOMENTA) and [FHM](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes). To erase texts from image, please refer to [OCR-SAM](https://github.com/yeungchenwa/OCR-SAM). The data file should look like this:


```
├── data
│   └── sampled_data
│       └── image
│           └── ori
│           └── erased
├── results
└── scripts
```

## Harmfulness Mining
Run harmfulness mining by:
```
cd scripts
python mining.py
```

## Model Scoring

First generate misbelief statement and reference answer by:

```
python gen_misb.py
```

Run model scoring by:
```
python scoring.py --exp_name exp_name --model_name model_name
```

## Iterative Refinement

Run iterative refinement by:

```
python refinement.py --exp_name exp_name --model_name model_name
```