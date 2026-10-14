# CV & NLP learner materials

Public learner notebooks and browser-based interactives for the Computer Vision and Natural
Language Processing course sequence, 6.4600.1x to 6.4600.5x. The current build includes the
Natural Language Processing foundations materials described below.

## Start here

Open notebooks from their course pages and make a copy in Google Drive before working. The setup
cells download each notebook's checked support files, so learners do not need to clone this
repository or download its data separately.

The course pages link to notebooks by their path in this repository, `codey-m/cv_nlp` on the `main`
branch, so renaming a notebook, moving it, or renaming the repository breaks its Colab button. The
links live in `CV_NLP/6.4600.*x/html/*_prac_text.html` and `*_lab_ov_text.html`.

## Current notebook build

| Notebook | Learner work | Course location |
|---|---|---|
| [`practice/m1_bpe_tokenizer.ipynb`](practice/m1_bpe_tokenizer.ipynb) | Build the BPE merge loop and encoder, then inspect why merge order matters | Module 1 coding practice after lecture 1.1, segment 4 |
| [`labs/lab1_text_representation.ipynb`](labs/lab1_text_representation.ipynb) | Compare word and BPE tokenization with the same bigram model; test whether perplexities across tokenizations are comparable | Lab 1: Text Representation and Language Models |
| [`practice/m2_beam_search.ipynb`](practice/m2_beam_search.ipynb) | Extend and prune a beam, trace two examples, and distinguish model likelihood from answer correctness | Module 2 beam-search coding practice; course placement follows lecture 2.1 segmentation |
| [`labs/lab2_language_model_systems.ipynb`](labs/lab2_language_model_systems.ipynb) | Evaluate prompting, LoRA, and quantization on an issued problem panel while auditing extraction, uncertainty, cost, and provenance | Lab 2: Language Model Systems |

The labs use a common assessment pattern: changed readiness cases test whether the learner's
functions transfer beyond notebook examples; an Evidence Record preserves the run; and later
checkpoints ask for a claim, its evidence and caveat, followed by the design of a next experiment.
Accuracy itself is not an execution requirement.

## Lab runtime notes

- **Lab 1** runs on a CPU. Learners implement four short functions, choose two tokenizers, and
  produce a comparison record. Its slowest cell takes under a minute.
- **Lab 2** requires a supported GPU session. Each learner receives one of twenty panel codes and
  runs prompting, LoRA, plain-int4 and calibrated-int4 on the same 200 problems. The panel
  fingerprint, function digest, stage provenance and four right-answer counts make that run
  auditable without requiring a target accuracy.
- **Lab 2 recovery** uses saved outputs when a session is interrupted or no GPU is present. Those
  records support inspection and interpretation, but are labeled as recovery evidence and do not
  satisfy the personal-run execution contract.
- **Lab 2 timing** is provisionally 12 to 15 T4 minutes for the required model stages. A fresh
  learner run on the supported service is still required to confirm that estimate.

Lab 2 loads its checked support bundle from `labs/data/lab2/`. Keep that directory and the notebook at their current
paths so the course links and notebook setup continue to work.

## Interactive course pages

The current build also publishes four focused interactives for 6.4600.1x:

- **Robot Theatre** makes the limitation of a bag-of-words reader visible.
- **Where the probability goes** traces probability mass in a count-based language model.
- **What text survives?** guides learners through a compact Unit 1 synthesis.
- **Order and visibility** builds geometric intuition for attention and token order.

The course loads the self-contained pages in [`widgets/`](widgets/) through GitHub Pages. Open
edX course assets no longer run their scripts, so moving or renaming this repository or that
directory would break the embedded activities. The generated [widget index](widgets/README.md)
links to the live pages and records their course locations.

## For maintainers

Do not edit these notebooks or Lab 2's support bundle by hand.
`CV_NLP/_workfiles/practice/build_practice.py` writes each notebook together with its solved
reference copy, and the reference copies stay in the private course repository. Nothing with a
solution in it belongs here.

Do not edit `widgets/` by hand either. `CV_NLP/docs/publish_widgets.py` copies the canonical
course assets here, rewrites the course iframes, and regenerates the widget index.
