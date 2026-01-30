---
layout: post
title: Paper Notes - Mechanistic?
date: 2026-01-30 10:40:16
description: Paper Notes for the `Mechanistic?` Paper
tags: paper-notes
categories:
---

Recently I've been really questionning the field of Interpretability, in terms of its potential future, potential for success and the differences it can make, so I've been thinking about it from a more fundamental level, mainly to improve my understanding from a meta pov as well as technical, so I thought I should do an exercise where I try to jot down my notes of a paper in my own language and share it across. These are my interpretations and notes on the paper, so I apologise in advance for any misinterpretation or mistakes.

First Paper I've chosen is a paper named [`Mechanistic?`](https://arxiv.org/abs/2410.09087). It takes about the origins of Mechanistic Interpretability as a term, a field of research and a community. So here we go...

<br>

# What Do We Really Mean by Mechanistic Interpretability?

Mechanistic Interpretability (MI) has become one of the most widely used—and most ambiguously defined—terms in modern AI interpretability research. Depending on who you ask, it can mean anything from causal reverse-engineering of neural networks to simply visualizing activations. This ambiguity is not accidental; it is the product of overlapping definitions, disciplinary histories, and community-driven norms.
In this post, I unpack where this confusion comes from, how MI differs (and overlaps) with NLP Interpretability (NLPI), and why these tensions matter for the future of the field.

## Four (and a Half) Definitions of Mechanistic Interpretability

The term mechanistic interpretability is currently overloaded. At least four distinct definitions circulate in the literature and community discourse:

1. **Narrow technical definition:** A research program focused on understanding neural networks through their causal mechanisms.
2. **Broad technical definition**: Any work that studies model internals—weights, activations, attention patterns, or intermediate representations.
3. **Narrow cultural definition**: Research originating from the MI community, historically associated with forums like Distill, LessWrong, or the AI Alignment Forum.
4. **Broad cultural definition:** Interpretability research in AI at large, especially in language models.
5. **Author-based definition (implicit)**: Compounding the confusion, “mechanistic interpretability” is sometimes used to describe who wrote a paper rather than what methods or goals it involves.

This definitional sprawl has diluted the original meaning of MI, particularly its grounding in causality.

## Mechanisms, Causality, and Abstraction

Mechanistic interpretability derives its name from causal mechanisms. A causal mechanism can be understood as a lawful transformation: a function that maps some subset of model variables (causes) to another subset (effects).

In neural networks, MI aims to discover the mechanisms that produce specific outputs from specific inputs, using intermediate representations as the primary level of abstraction.

However, real-world causal systems—biological, physical, or artificial—are deeply complex. Multiple pathways can lead to the same outcome, making full causal explanations unwieldy. Human explanations are one way to manage this complexity: they distill explanations down to the most proximal or salient mechanisms.

Neural networks offer a different option. Since we are not limited to human-scale explanations, MI often relies on causal abstraction—distilling complex causal structure into interpretable, tractable forms without restricting ourselves to only the most local mechanisms.

## From Distill to “Reverse Engineering Neural Networks”

The term mechanistic interpretability was coined by Chris Olah and first publicly used in posts on [Distill.pub](https://distill.pub/). Early work focused heavily on understanding how weights implement mechanisms, particularly in vision models.

The phrase “reverse engineering neural networks” gained popularity later, especially after the Distill team moved to Anthropic and began publishing [transformer circuit](https://transformer-circuits.pub/) threads. The term became so dominant that even the [ICML 2024 MI Workshop](https://icml2024mi.pages.dev/) adopted it to define its scope.

Ironically, much contemporary MI work no longer explicitly references causal mechanisms at all. Today, almost any form of internal inspection—causal or not—is often labeled “mechanistic interpretability.”

## The NLPI–MI Divide

A major theme in this discussion is the tension between two communities:

- **NLPI (NLP Interpretability):** Active since early LSTM and RNN models, publishing primarily in ACL and related venues.
- **MI (Mechanistic Interpretability):** Emerging largely from computer vision and alignment-oriented communities, later pivoting to NLP.

One of the major points of tension was that, rather than building on NLPI’s extensive prior work or engaging with them, MI community started working on interpretability research independently, often reinventing earlier findings and repeating epistemological debates around:

- Correlation vs. causation
- Simple features vs. complex subnetworks
- Expressive mappings vs. constrained interpretations

This duplication was not purely technical, it was cultural.

Below discussed are some of the technical avenues where such duplication can be seen.

### Vector Semantics: Old Lessons, New Names

The explosion of vector semantics after word2vec sparked extensive work on interpreting embeddings. Many modern ideas like task vectors, steering vectors, additive representation properties, trace their lineage to this era.

But early critics showed that embedding structure often reflects word frequency rather than deep semantic meaning. Many of these critiques remain highly relevant today as well, especially for correlational interpretability methods like similarity metrics and activation comparisons.

### Neurons, Circuits, and Polysemanticity

Both NLPI and MI have attempted to localize concepts within models:

- NLPI localized linguistic phenomena to neurons or layers.
- MI pursued neuron-level explanations and later circuits—subgraphs of neurons responsible for specific behaviors.

Single-neuron analysis, however, has faced sustained criticism:

- Large models cannot be reduced to a sum of independent parts.
- Behavioral composition is often non-linear.
- Polysemanticity, where a neuron responds to multiple unrelated concepts, makes interpretation ambiguous.

MI community tried to solve polysemanticity, one of the example was using sparse autoencoders (SAE), which still relies on the assumption of linearity and naturally emerging sparsity, but like earlier neuron analysis methods, it requires expensive causal validation.

### Probing: A Familiar Story

Probing methods originated in NLPI as a way to extract linguistic information from hidden states. They were criticized for weak baselines, and lot of times, probes were extracting more information from randomly initialized vectors than full trained networks

Modern MI work often uses linear probes to project model states into interpretable subspaces, but these approaches inherit the same limitations:

- Weak causal grounding
- Sensitivity to probe capacity
- Ambiguity about what is truly encoded vs. what the probe learns

## Cultural, Not Just Methodological, Differences

During the early days, machine-learning in general was primarily attributed to computer vision research and lot of interpretability works were being done in CV only, shaping the methods NLP researchers adopted. The MI term itself was first used in CV contexts before migrating to NLP after breakthroughs like GPT-3.

The deeper divide, however, was cultural:

- MI research initially lived on blogs, forums, Slack, and Discord.
- NLPI research lived in peer-reviewed venues like ACL.

The absence of NLPI researchers from online forums was misread as disinterest, when it was largely a difference in publication norms.

When MI researchers began publishing in conferences, they faced criticism for limited engagement with prior NLPI literature. Despite this, MI rapidly attracted resources, attention, and prestige, leading many NLPI researchers to adopt MI terminology.

# “Everyone Is Mechanistic Now”

Today, most interpretability researchers identify, at least loosely, with MI. Funding and visibility have accelerated this shift, but terminology alone has not unified the field.

Despite methodological overlap, differences in norms, goals, and epistemology remain.

# Why This Still Matters

For all the confusion and tension it introduced, the MI community has played a major role in revitalizing interpretability research. Its energy, resources, and ambition have expanded the field’s reach.

At the same time, many MI researchers are motivated by AI alignment concerns, where the long-term value of MI is actively debated. It is possible that alignment priorities will eventually shift away from MI, taking some resources with them.

But interpretability will persist.

NLPI and MI researchers share too much in common, like scientific curiosity, social responsibility, and a commitment to understanding powerful models,for this work to disappear. As long as our models remain opaque, we will keep trying to open them up.

And in that sense, the question is not whether mechanistic interpretability survives, but what we ultimately decide it means.
