# CV & NLP notebooks

Learner notebooks for the Computer Vision and Natural Language Processing courses, 6.4600.1x to
6.4600.5x, opened from the course pages in Google Colab.

Open a notebook from its course page, then make a copy in Drive before working. You do not need to clone this
repository or download its data files.

The course pages link to notebooks by their path in this repository, `codey-m/cv_nlp` on the `main`
branch, so renaming a notebook, moving it, or renaming the repository breaks its Colab button. The
links live in `CV_NLP/6.4600.*x/html/*_prac_text.html` and `*_lab_ov_text.html`.

| Notebook | Course page |
|---|---|
| `practice/m1_bpe_tokenizer.ipynb` | Coding practice after lecture 1.1, segment 4 |
| `labs/lab1_text_representation.ipynb` | Lab 1: Text Representation and Language Models |
| `practice/m2_beam_search.ipynb` | Coding practice after lecture 2.1, beam search; page drafted, placed once 2.1 is segmented |
| `labs/lab2_language_model_systems.ipynb` | Lab 2: Language Model Systems |

## Lab runtime notes

- **Lab 1** runs on a CPU. Its slowest cell takes under a minute.
- **Lab 2** requires a supported GPU session. Each learner receives a code for a 200-problem panel and runs the
  prompting, LoRA, plain-int4 and calibrated-int4 conditions on those same problems. Saved outputs remain available
  when a session is interrupted, but they are labeled as recovery evidence and do not satisfy the execution contract.
  Scaling the measured full-record stages to the 200-problem panel gives a provisional 12 to 15 T4 minutes; a fresh
  learner run must confirm that estimate. Course instructions will name the supported service.

Lab 2 loads its checked support bundle from `labs/data/lab2/`. Keep that directory and the notebook at their current
paths so the course links and notebook setup continue to work.

## For maintainers

Do not edit these notebooks or Lab 2's support bundle by hand. `CV_NLP/_workfiles/practice/build_practice.py` writes
each notebook together with its solved reference copy, and the reference copies stay in the private course repository.
Nothing with a solution in it belongs here.
