# **Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings**   
   
This is an official repo of the paper "**Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings**," which is accepted to ICASSP2025.   

**Abstract**：This paper presents a causal speech enhancement (SE) model based on a two-branch complex nested U-Net (CNUNet-TB) architecture combined with a two-stage (TS) training method that leverages speech embeddings from a large self-supervised speech representation learning (SRL) model. The proposed architecture enhances performance by simultaneously estimating complex masks and the speech spectrum, effectively handling complex speech data. The SE model is initially trained by fusing the self-supervised speech embeddings with the model’s latent vectors, which are then stored for the second stage of training. In the second stage, the SE model is trained to replicate these latent vectors without relying on the SRL model, ensuring causality and eliminating the need for the SRL model during inference. Experimental results demonstrate that the proposed CNUNet-TB-TS effectively replicates the stored latent vectors with strong speaker representation, achieving superior performance compared to recent causal SE models.

## Update:  
* **2024.09.24** Upload codes  
* **2024.10.14** Upload demo samples  

## Requirements 
This repo is tested with Ubuntu 22.04, PyTorch 2.0.4, Python3.10, and CUDA12.2. For package dependencies, you can check: requirements.txt  


## Getting started    
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([options.py](https://github.com/seorim0/SE-using-SRL-Model/blob/main/options.py)) 
```   
# dataset path
noisy_dirs_for_train = '../Dataset/train/noisy/'   
noisy_dirs_for_valid = '../Dataset/valid/noisy/'   
```   
* You need to modify the `find_pair` function in [utils](https://github.com/seorim0/SE-using-SRL-Model/blob/main/utils/progress.py) according to the data file name you have.        
* You can simply change any parameter settings if you need to adjust them.   
3. Run [train_interface.py](https://github.com/seorim0/SE-using-SRL-Model/blob/main/train_interface.py)

## Results and Analysis
![1](https://github.com/user-attachments/assets/48fb343e-56ad-46d1-9ee7-4ef75d0c8286)
- First, converting NUNet to a complex version (CNUNet) yields significant performance improvements. Additionally, the two-branch version of CNUNet (CNUNet-TB) improves all metrics compared to the single-branch version (CNUNet-SB), though the improvement was marginal. The incorporation of WavLM Large’s speech embeddings in stage 1 significantly improved performance across all metrics. When the speech embeddings were replicated in Stage-2, where causality is enforced and the SRL model is removed during inference, the model still achieved impressive results.

![2](https://github.com/user-attachments/assets/cbf9cc6f-33ae-4b73-84ce-54495f815169)
- Recent SE models effectively remove background noise (measured via BAK), but tend to over-suppress, leading to speech distortion and even lower signal clarity (SIG) scores than the original noisy speech. However, our proposed model significantly improved the SIG score while enhancing the BAK score also, resulting in an overall improvement (OVL). These results confirm that the proposed model produces enhanced speech perceived as both natural and intelligible, even in causal settings.

![3](https://github.com/user-attachments/assets/fb89886f-8e29-46ea-97b6-a30bc029a05d)
- Our CNUNet-TB-TS achieved the highest PESQ score, and intelligibility-related metrics are significantly higher for our model compared to others.

![4](https://github.com/user-attachments/assets/3f17fcf9-08c4-40d1-b281-a2ad6f6c41d1)
- We visualized the latent vectors using t-SNE, based on 384 utterances from 14 male and 14 female speakers among the training dataset. From this, two key observations are possible. First, when self-supervised speech embeddings are used (Stage-1), there is a clear separation between the speaker’s gender and some distinction between speakers, resulting in improved SE performance. This separation is likely due to WavLM Large, trained on a large-scale audio dataset, producing embeddings with strong speaker representations. Also, these results align with previous observations in [[ref]](https://arxiv.org/pdf/2302.11558), where injecting speaker gender embeddings into the SE model improves the SE performance. Second, this separation and clustering are still maintained even after Stage-2 training. This shows that our training strategy successfully retains the benefits of self-supervised speech embeddings in a causal SE model, leading to high signal clarity (CSIG and SIG in Tables I and II).


## Demo
More demo samples can be found [here](https://github.com/seorim0/SE-using-SRL-Model/blob/main/demo/).  

- Clean  

https://github.com/user-attachments/assets/91f1d067-184c-4035-8ea0-a3c08c1e93c0



- Noisy  

https://github.com/user-attachments/assets/737f82cd-bb07-4f20-ab21-70f9a870c9fc



- NUNet-TLS  

https://github.com/user-attachments/assets/faffda76-62e0-4ba1-a1d6-161e9ffe0449


- Proposed (without two-stage learning)  

https://github.com/user-attachments/assets/8d747c75-4578-4fbd-8d7b-fb3377a70918


- Proposed  

https://github.com/user-attachments/assets/abdeaac9-f4e3-418f-9c1c-7d2b9b2c4cfb

  
 
## References   
**Monoaural Speech Enhancement Using a Nested U-Net with Two-Level Skip Connections**   
S. Hwang, S. W. Park, and Y. Park   
[[paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/hwang22b_interspeech.pdf)  [[code]](https://github.com/seorim0/NUNet-TLS)   


## Contact  
Please get in touch with us if you have any questions or suggestions.   
E-mail: allmindfine@yonsei.ac.kr
