# Task-Specific Machine Reading Comprehension and Retrospective Reader - CS492 NLP Project Team Zoommalgo
Question Answering task using NAVER KorQuAD-Open dataset.

## Path to Best Pretrained Model
* kaist002/korquad-open-ldbd3/273/electra_gs16000_e1

## Train in NSML
```bash
sh run_nsml.sh
```

## Train in Local

```bash
sh run_local.sh
```
## Dataset Information
NAVER KorQuAD-Open training dataset is consisted of 66367 questions with 18.47 paragraphs per question in average. Among these questions, 64.6% are questions which includes the answer in multiple paragraphs. The goal of this task is to give the correct answer for a question if the answer is included in paragraphs. On the other hand, if the answer does not exist in context paragraphs, the machine has to predict null string for this case.

## Methods

Main contribution is from changing the answer prediction method. Giving high priority to answer containing predictions than null string prediction gave a huge boost to the performance. 

See the codes:
- https://github.com/qbhan/cs492i_nlp/blob/main/open_squad_metrics.py#L448

Instead of multilingual BERT, KoELECTRA is used for faster training and better test f1 accuracy. (monologg/koelectra-base-v2-discriminator).
Instead of using three paragraphs, using four paragraphs lead to a slight improvement to the test f1 accuracy.

See the codes:
- https://github.com/qbhan/cs492i_nlp/blob/main/open_squad.py#L644
- https://github.com/qbhan/cs492i_nlp/blob/main/open_squad.py#L650

Other than these contributions, we tried several approaches; applying retrospective reader with sketchy and intensive reader, trying different paragraph extraction method based on similarity between question and context (force answer, max 2, sentence transformer). Also, in the process of some preliminary experiments, we found that there is a huge discrepancy between valid and test accuracy. This discrepancy came from using different criteria for calculating the accuracy. For validation metric, the score was calculated per paragraphs while for test metric, it was calculated per question. For the ease of debugging, we changed the metric of validation similar to test metric.

## Results
* After fintuning with appropriate hyperparameters, our best test f1 score was 0.703. We ranked second place in <a href= "https://ai.nsml.navercorp.com/">NSML</a> leaderboard system.


| Model                   | Test F1 score |
| -----------------------:| -------------:|
| Baseline Model          | 0.361         |
| Ours                    | **0.703**     | 

## References
* <a href = "https://arxiv.org/pdf/2001.09694">Retrospective reader for machine reading comprehension (2020)</a>
* <a href = "https://openreview.net/pdf?id=r1xMH1BtvB">ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (2020)</a>

