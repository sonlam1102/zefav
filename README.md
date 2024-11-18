# ZeFaV: Boosting Large Language Models for Zero-shot Fact Verification
Source code for ZeFaV in the paper "ZeFaV: Boosting Large Language Models for Zero-shot Fact Verification"

Author: Son T. Luu, Hiep Nguyen, Trung Vo and Le-Minh Nguyen 

How to run: 
1. The prompt.py: Contains the main source code for ZeFaV prompting based on LLama-3.0 13B 
2. The "infore" folder: Containing the source code for reorganizing information by InfoRE
3. The "llama-relation-train" folder: Contains the source code for fine-tuning LLama3.0 7B for the Relation Extraction task based on the FewRel dataset

Note: You must use your access token in HuggingFace to access the LLama model. Please find the guide here: https://huggingface.co/docs/hub/en/security-tokens 

If you find it difficult to reproduce the source code, we have the available result in the "dataset" folder.  

Link to the publication: https://link.springer.com/chapter/10.1007/978-981-96-0119-6_28  

Please cite our paper if you use ZeFaV: 
```
@InProceedings{10.1007/978-981-96-0119-6_28,
author="Luu, Son T.
and Nguyen, Hiep
and Vo, Trung
and Nguyen, Le-Minh",
editor="Hadfi, Rafik
and Anthony, Patricia
and Sharma, Alok
and Ito, Takayuki
and Bai, Quan",
title="ZeFaV: Boosting Large Language Models for Zero-Shot Fact Verification",
booktitle="PRICAI 2024: Trends in Artificial Intelligence",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="288--295",
abstract="In this paper, we propose ZeFaV - a zero-shot based fact-checking verification framework to enhance the performance on fact verification task of large language models by leveraging the in-context learning ability of large language models to extract the relations among the entities within a claim, re-organized the information from the evidence in a relationally logical form, and combine the above information with the original evidence to generate the context from which our fact-checking model provide verdicts for the input claims. We conducted empirical experiments to evaluate our approach on two multi-hop fact-checking datasets including HoVer and FEVEROUS, and achieved potential results results comparable to other state-of-the-art fact verification task methods.",
isbn="978-981-96-0119-6"
}


```
