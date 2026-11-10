# LLM-AggreFact: licences of the datasets and possible out-of-domain use

Checked on 2026-11-09. Sources: the Hugging Face card of the collection, the GitHub API
(licence detected on each repository), the licence files and READMEs of the original
repositories, the Hugging Face API of the source datasets, the MiniCheck paper (arXiv
2404.10774). This is a reading of the licences as displayed, not legal advice.

## The collection

`lytang/LLM-AggreFact` on Hugging Face.

- Declared licence: CC BY-ND 4.0. No redistribution of a modified version: a conversion to the
  RAGnarok format stays local.
- Gated access. One has to accept, with a Hugging Face account, that the resource "is
  permitted for use as an evaluation benchmark". Terms accepted on 2026-11-09. The token
  stored on the measurement machine must also allow reading public gated repositories,
  otherwise downloading the files returns a 403 error.
- The card states: "Data in the benchmark should not be used in pretraining or fine-tuning
  any NLP models."
- 11 datasets since August 2024: RAGTruth was added, converted, with the claims judged
  unverifiable removed.
- Published protocol: balanced accuracy per dataset, threshold at the middle of the score
  scale (0.5), no per-dataset tuning in the main results. Each sentence is checked on its
  own. Dev and test splits taken from the original datasets for AggreFact, TofuEval, WiCE and
  ClaimVerify, drawn at random in equal shares per query for the others.

## Dataset by dataset

| Dataset | Original repository | Licence of the annotations | Source documents | Level |
|---|---|---|---|---|
| WiCE | `ryokamoi/wice` | ODC-BY 1.0 (code MIT) | Wikipedia (CC BY-SA) and pages archived by Common Crawl | 1 |
| REVEAL | `google/reveal` (Hugging Face, gated) | CC BY-ND 4.0 | Wikipedia | 1 |
| ClaimVerify | `nelson-liu/evaluating-verifiability-in-generative-search-engines` | MIT | web pages cited by generative search engines | 1 |
| FactCheck-GPT | `yuxiaw/Factcheck-GPT` | Apache-2.0 (repository; no separate licence for the data) | web search results | 1 |
| ExpertQA | `chaitanyamalaviya/ExpertQA` | MIT | web pages | 1 |
| TofuEval-MediaS | `amazon-science/tofueval` | MIT-0 | MediaSum, NPR and CNN transcripts: "restrict your usage of this dataset to research purpose only" | 2 |
| TofuEval-MeetB | `amazon-science/tofueval` | MIT-0 | MeetingBank: CC BY-NC-SA 4.0 according to its Hugging Face card | 2 |
| AggreFact-CNN | `Liyan06/AggreFact` | no licence in the repository | CNN/DailyMail articles | 3 |
| AggreFact-XSum | `Liyan06/AggreFact` | no licence in the repository | BBC articles (XSum, licence "unknown" on Hugging Face) | 3 |
| LFQA | `timchen0618/LFQA-Verification` | no licence in the repository | retrieved web documents | 3 |
| RAGTruth | `ParticleMedia/RAGTruth` | MIT | CNN/DailyMail, MS MARCO, Yelp | excluded from the out-of-domain score |

Levels:

1. Explicit licence on the annotations, with no restriction of use beyond attribution or the
   absence of derivatives. Usable.
2. Annotations under a permissive licence, documents restricted to research or
   non-commercial use. Usable for a non-commercial research evaluation.
3. No explicit licence on the annotations or on the documents. Use rests only on the access
   terms of the collection, which allow evaluation.

## Retained (decision of 2026-11-09)

- **Out-of-domain decision set**: the five level 1 datasets, WiCE, REVEAL, ClaimVerify,
  FactCheck-GPT and ExpertQA. The out-of-domain target of `TARGETS.md` bears on these five
  datasets and on them only.
- **Not retained**: the two TofuEval datasets (level 2) and the three level 3 datasets.
  AggreFact-CNN was not out of domain for the documents anyway, since RAGTruth Summary also
  draws on CNN/DailyMail.
- **Publication**: under the CC BY-ND licence, the converted data are never published, in a
  repository or on Hugging Face, even reformatted. Only metrics are published. Same rule as
  for RAGTruth.
- **Use**: no use for training or selection; the dev split of the collection is not used; a
  check of document overlap with the training set of RJ is made before the evaluation.

Reference points of the public leaderboard for MiniCheck-Flan-T5-Large, balanced accuracy:
WiCE 72.2, REVEAL 86.2, ClaimVerify 74.6, FactCheck-GPT 74.7, ExpertQA 59.0. HHEM-2.1-open is
listed there as well; its figures are in `ood/published_bacc.json`.

## Reference models

- HHEM-2.1-Open: Apache-2.0. Its card does not describe the training data; it reports results
  on AggreFact-SOTA, RAGTruth-Summ and RAGTruth-QA.
- MiniCheck (code): Apache-2.0. Trained on 21K ANLI examples and 14K synthetic examples
  according to the paper; whether the dev split of LLM-AggreFact was used for selection is not
  documented in the part read.

## Sources

- https://huggingface.co/datasets/lytang/LLM-AggreFact
- https://github.com/Liyan06/MiniCheck
- https://arxiv.org/html/2404.10774v2
- https://llm-aggrefact.github.io/
- https://huggingface.co/vectara/hallucination_evaluation_model
- https://github.com/ryokamoi/wice
- https://huggingface.co/datasets/google/reveal
- https://github.com/nelson-liu/evaluating-verifiability-in-generative-search-engines
- https://github.com/yuxiaw/Factcheck-GPT
- https://github.com/chaitanyamalaviya/ExpertQA
- https://github.com/amazon-science/tofueval
- https://github.com/zcgzcgzcg1/MediaSum
- https://huggingface.co/datasets/huuuyeah/meetingbank
- https://github.com/Liyan06/AggreFact
- https://github.com/timchen0618/LFQA-Verification
- https://github.com/ParticleMedia/RAGTruth
