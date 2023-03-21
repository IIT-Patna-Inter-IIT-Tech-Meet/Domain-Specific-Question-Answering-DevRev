# Domain Specific Question Answering
[Problem Statement](https://github.com/rdev12/Domain-Specific-Question-Answering-DevRev/blob/master/Problem_Statement.pdf)

## Abstract
The report outlines our pipeline for domain-specific question answering in a open-QA
setting. We have created domain-adaptable rankers fine-tuned using knowledge
distillation in order to re-rank the passages retrieved using BM25. We propose a novel
difficulty prediction heuristic which dynamically determines the number of paragraphs to
be fed to the reader by utilising the ranker scores and the remaining time. Finally, we use
signals from reader, ranker as well as the retriever to determine the answerability of the
question.

[End-term Report](https://github.com/rdev12/Domain-Specific-Question-Answering-DevRev/blob/master/End-term%20Report.pdf)
[Mid-term Report](https://github.com/rdev12/Domain-Specific-Question-Answering-DevRev/blob/master/Mid-term%20Report.pdf)
[Presentation](https://github.com/rdev12/Domain-Specific-Question-Answering-DevRev/blob/master/Presentation.pdf)
