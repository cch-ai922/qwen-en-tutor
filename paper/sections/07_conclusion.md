# 7. Conclusion

We presented a data-generation pipeline for English-tutor dialogue
that goes beyond a flat "good behavior" corpus to systematically
cover the long tail of tutor-side behavior. The pipeline contributes
a 12-stream SFT taxonomy decomposing tutor behavior by violation
axis; a 4-variant structural design for persistent abuse handling
that defeats the positional-shortcut failure mode; a yield-aware
top-up loop with declarative ratio targets; locale-aware prompt
engineering with strict-Latin and allow-L1 variants; and a
six-filter cascade whose LLM-judge component we audit and patch.

Empirically, we demonstrate the pipeline at the smallest practical
scale (a 0.8B-parameter student trained on a single RTX 3060
12GB consumer GPU), and show that the resulting model on the
mechanical sentinel-firing metric sharply outperforms a same-base
ablation trained without the 4-variant persistent design, on
redirect-axis F1 sharply outperforms a same-size off-the-shelf
instruct model, and on naturalness approaches the 9B-teacher
distillation upper bound.

Future work includes (i) multi-locale empirical evaluation extending
beyond `china` to `japan`/`italy`/etc.; (ii) larger-student
experiments to measure how each contribution scales with student
capacity; (iii) a learned naturalness filter to close the C1 / C2
naturalness gap; and (iv) multi-family distillation studies (e.g.
Llama-3 student from a Qwen3.5 teacher) to test the intra-family
distillation assumption embedded in our setup.
