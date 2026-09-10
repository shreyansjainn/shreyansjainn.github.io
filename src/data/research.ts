// Ongoing research directions + open-source software.

export const directions = [
  {
    title: 'Training dynamics',
    body: 'When mechanisms form: whether behavioural properties develop gradually or through qualitative transitions, and what that trajectory reveals about how models acquire, organise, and retain behavioural structure.',
  },
  {
    title: 'Behavioural decomposition',
    body: 'Breaking complex behaviours — sycophancy, deception — into their constituent mechanisms, testing whether they recompose, and identifying the distinct modes a single named behaviour can take.',
  },
];

// Lighter-touch threads I'm actively curious about.
export const additionalInterests = [
  'Geometry-aware steering',
  'Manifold geometry',
  'Multilingual interpretability',
];

// Schematic illustration of the decomposition thesis (weights are illustrative,
// not measured results) — a named behaviour as a composition of atomic traits.
export const decomposition = {
  behaviour: 'sycophancy',
  source: {
    label: 'Gotta Catch Them All: The Modes of Sycophancy',
    href: 'https://arxiv.org/abs/2607.20146',
  },
  parts: [
    { name: 'agreeableness', weight: 34 },
    { name: 'deference', weight: 26 },
    { name: 'praise-seeking', weight: 22 },
    { name: 'answer-conformity', weight: 18 },
  ],
};

export const software = [
  {
    name: 'K-Steering',
    body: 'Non-linear, multi-attribute steering framework for language models.',
    href: 'https://github.com/withmartian/k-steering',
  },
  {
    name: 'visualizing-training',
    body: 'Hidden Markov model toolkit for visualising neural network training dynamics (ICLR 2025 Blogpost Track).',
    href: 'https://github.com/shreyansjainn/visualizing-training',
  },
];
