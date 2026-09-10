// CV data, sourced from CV-Shreyans Jain.pdf. Keep the PDF and this in sync.

export type Role = {
  org: string;
  title: string;
  period: string;
  detail?: string; // e.g. "with Amirali Abdullah"
  points?: string[];
};

export const summary =
  'Independent interpretability researcher working on how coherent, high-level behaviours in language models arise from distributed internal representations, and whether they decompose into identifiable mechanisms. My working view is that behaviours we name at the surface — sycophancy, deception — are compositions of smaller measurable traits, and that tracking when those components form during training reveals how models acquire behavioural structure. Before interpretability research, I spent 8 years building production machine learning systems.';

export const researchExperience: Role[] = [
  {
    org: 'London AI Safety Research (LASR) Labs',
    title: 'Research Fellow',
    period: 'Jul 2026 – Present',
    detail: 'with Satvik Golecha (UK AI Security Institute — Model Transparency Team)',
    points: [
      'Developing the science of model organisms — systematically studying persona implantation and behavioural evaluation techniques, in collaboration with the UK AI Security Institute.',
      'Investigating the training dynamics and underlying mechanisms of emergent misalignment in language models.',
    ],
  },
  {
    org: 'Thoughtworks',
    title: 'Research Intern',
    period: 'Apr 2026 – Jul 2026',
    detail: 'with Amirali Abdullah',
    points: [
      'Decomposed sycophancy into compositions of atomic psychometric traits, characterising distinct modes of sycophantic behaviour (under review).',
    ],
  },
  {
    org: 'Martian',
    title: 'Research Engineer Fellow',
    period: 'Dec 2025 – Mar 2026',
    detail: 'with Narmeen Oozeer and Phillip Quirke',
    points: [
      'Developed K-Steering, a first-of-its-kind non-linear steering framework for multi-attribute control of language models, released as an open-source package.',
      'Built a task-agnostic LLM-judge prompt optimisation system that infers task definitions and rubrics directly from human feedback signals.',
    ],
  },
  {
    org: 'Thoughtworks',
    title: 'Research Fellow',
    period: 'Aug 2025 – Oct 2025',
    detail: 'with Amirali Abdullah',
    points: [
      'Built a psychometric evaluation framework for LLMs in law enforcement contexts, integrating HEXACO, synthetic situational judgment tests, and psychologically derived synthetic personas.',
    ],
  },
  {
    org: 'Independent Research',
    title: 'Independent Researcher',
    period: 'Jun 2024 – Present',
    detail: 'with Naomi Saphra, Amirali Abdullah, and Shivam Raval',
    points: [
      'Developed a framework for visualising training dynamics in neural networks using hidden Markov models, released as an open-source package (ICLR 2025 Blogpost Track).',
      'Analysed the factors affecting the geometry of features in superposition in toy models, and the effect of non-uniform feature sparsity on superposition.',
    ],
  },
];

export const industryExperience: Role[] = [
  {
    org: 'Physarum',
    title: 'Reinforcement Learning Engineer (part-time, contract)',
    period: 'Oct 2024 – Aug 2025',
    points: [
      'Built a next-best-action recommender using non-contextual bandits and a base-price optimisation system using reinforcement learning for a British telecoms company.',
    ],
  },
  {
    org: 'GEP',
    title: 'Senior Data Scientist',
    period: 'Jul 2021 – May 2024',
    points: [
      'Led the P2P AI pod (invoice OCR, search, requisition, catalog) product development and a team of five data scientists, serving hundreds of thousands of users daily.',
    ],
  },
  {
    org: 'BookMyShow',
    title: 'Data Scientist II',
    period: 'Nov 2019 – Jul 2021',
    points: [
      'Built a real-time personalisation system combining affinity (GloVe), collaborative (ALS), content (Doc2Vec) and exploration (RL) signals, plus an end-to-end dynamic pricing and demand forecasting platform.',
    ],
  },
  {
    org: 'Hotstar',
    title: 'Data Scientist',
    period: 'Jan 2019 – Oct 2019',
    points: ['Churn prediction, user segmentation, topic modelling, and text classification.'],
  },
  {
    org: 'Fractal Analytics',
    title: 'Consultant',
    period: 'Mar 2016 – Jan 2019',
    points: ['Recommender systems, demand forecasting, and text mining models for Fortune 500 clients.'],
  },
];

export const education = [
  {
    title: 'B.Tech in Civil Engineering',
    org: 'Malaviya National Institute of Technology, Jaipur',
    period: '2011 – 2015',
  },
  {
    title: 'AI Safety Fundamentals (Alignment)',
    org: 'BlueDot Impact',
    period: 'Nov 2024',
  },
];

export const selectedWriting = [
  {
    title: 'Effects of Non-Uniform Sparsity on Superposition in Toy Models',
    venue: 'LessWrong',
    href: 'https://www.lesswrong.com/posts/WwxG8RRHrorJgpoAk/effects-of-non-uniform-sparsity-on-superposition-in-toy',
  },
  {
    title: 'Geometry of Features in Superposition in Toy Models',
    venue: 'BlueDot Impact Capstone Project',
    href: '/blog/pentagon-feature-geometry',
  },
];
