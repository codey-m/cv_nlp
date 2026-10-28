# Multimodal AI widgets

The interactive widgets in the Multimodal AI courses on MIT Learn. The courses load each widget from
this folder through GitHub Pages, so every link below opens the exact page learners see inside
the lesson.

## 6.4600.1x Multimodal AI: Language Model Foundations

| Widget | Where | What it shows |
| --- | --- | --- |
| [Robot Theatre](https://codey-m.github.io/cv_nlp/widgets/widget-separate-by-hand.html) | Unit 1, Lec. 1: Tokenization & BoW Classifiers | A director that reads words but not their order |
| [Where the probability goes](https://codey-m.github.io/cv_nlp/widgets/widget-where-the-probability-goes.html) | Unit 1, Lec. 2: Count based language models |  |
| [What text survives?](https://codey-m.github.io/cv_nlp/widgets/unit-summary-what-text-survives.html) | Unit 1, Lec. 2: Count based language models | Unit 1 synthesis: what text survives? |
| [Order and visibility](https://codey-m.github.io/cv_nlp/widgets/widget-order-and-visibility.html) | Unit 2, Lec. 4: Attention and transformers |  |
| [How context reaches a prediction](https://codey-m.github.io/cv_nlp/widgets/unit-summary-how-context-reaches-prediction.html) | Unit 2, Lec. 4: Attention and transformers | Unit 2 synthesis: how context reaches a prediction |

## 6.4600.5x Multimodal AI: Systems and Engineering

| Widget | Where | What it shows |
| --- | --- | --- |
| [Crossfade](https://codey-m.github.io/cv_nlp/widgets/widget-crossfade.html) | Unit 1 overview | Crossfade: combining picture and sound |
| [How do a picture and a sentence meet?](https://codey-m.github.io/cv_nlp/widgets/unit-summary-picture-and-sentence.html) | Unit 1, Lec. 4: Multimodal reasoning | Unit 1 synthesis: how do a picture and a sentence meet? |
| [Stunt School](https://codey-m.github.io/cv_nlp/widgets/widget-stunt-school.html) | Unit 2 overview | Stunt School: learning a trick from rewards |
| [What is it actually rewarded for?](https://codey-m.github.io/cv_nlp/widgets/unit-summary-what-is-it-rewarded-for.html) | Unit 2, Lec. 6: Deployment, evaluation, and frontiers | Unit 2 synthesis: what is it actually rewarded for? |

## For course developers

[Visual methods catalog](https://codey-m.github.io/cv_nlp/widgets/visual-methods.html): fourteen methods with sample visuals,
reader actions, evidence requirements and source examples. Use it to design explainers,
unit summaries and views of lab evidence.

## Review tips

- Each widget is one self-contained HTML file. It makes no network requests, so a downloaded copy works too.
- In the course each widget sits in a column about 880 pixels wide. Narrow the browser window to that width to see the layout learners get.

## How this folder works

Nothing here is edited by hand. Each widget is developed in the course repository, and that
repository's `docs/publish_widgets.py` copies it here and rewrites this README. A push to `main`
updates the live widget a minute or two later, with no course import. The repository name is part
of every widget address, so it must never change.

The developer catalog has its own source at `CV_NLP/docs/visual-methods.html`; the same
publisher copies it here. It is a standalone adaptation of the Visual Stories methods catalog.
