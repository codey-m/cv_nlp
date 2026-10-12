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
- **Lab 2** has a complete graded path without a GPU: it uses saved model outputs for the full-model comparisons.
  Select a T4 GPU in Colab if you want to run its live fine-tuning and quantization demonstrations; the notebook
  gives a pilot estimate of about 35 minutes of GPU compute.

Lab 2 loads its checked support bundle from `labs/data/lab2/`. Keep that directory and the notebook at their current
paths so the course links and notebook setup continue to work.

## For maintainers

Do not edit these notebooks or Lab 2's support bundle by hand. `CV_NLP/_workfiles/practice/build_practice.py` writes
each notebook together with its solved reference copy, and the reference copies stay in the private course repository.
Nothing with a solution in it belongs here.
