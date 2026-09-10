// Publications, reverse-chronological. Add a new entry at the top.
// `me: true` bolds your name; `eq: true` marks equal contribution (*).
// `selected: true` surfaces it in the home-page highlight.

export type Link = { label: string; href: string };

export type Author = { name: string; me?: boolean; eq?: boolean };

export type Publication = {
  title: string;
  authors: Author[];
  venue: string;
  year: number;
  status?: string; // e.g. "Under review", "Findings", "Extended Abstract"
  selected?: boolean;
  links?: Link[];
};

export const publications: Publication[] = [
  {
    title: 'Gotta Catch Them All: The Modes of Sycophancy',
    authors: [
      { name: 'Shreyans Jain', me: true },
      { name: 'Alexandra Yost' },
      { name: 'Amirali Abdullah' },
    ],
    venue: 'Under review',
    year: 2026,
    status: 'Under review',
    selected: true,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2607.20146' }],
  },
  {
    title:
      'Measure What Matters: Psychometric Evaluation of AI with Situational Judgment Tests',
    authors: [
      { name: 'Alexandra Yost', eq: true },
      { name: 'Shreyans Jain', me: true, eq: true },
      { name: 'Shivam Raval' },
      { name: 'Grant Corser' },
      { name: 'Allen Roush' },
      { name: 'Nina Xu' },
      { name: 'Jacqueline Hammack' },
      { name: 'Ravid Shwartz-Ziv' },
      { name: 'Amirali Abdullah' },
    ],
    venue: 'EMNLP 2026 (Findings)',
    year: 2026,
    selected: true,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2510.22170' }],
  },
  {
    title:
      'Beyond Linear Steering: Unified Multi-Attribute Control for Language Models',
    authors: [
      { name: 'Narmeen Oozeer' },
      { name: 'Luke Marks' },
      { name: 'Shreyans Jain', me: true },
      { name: 'Fazl Barez' },
      { name: 'Amirali Abdullah' },
    ],
    venue: 'EMNLP 2025 (Findings)',
    year: 2025,
    links: [
      { label: 'arXiv', href: 'https://arxiv.org/abs/2505.24535' },
      { label: 'code', href: 'https://github.com/withmartian/k-steering' },
    ],
  },
  {
    title: 'How to Visualize Training Dynamics in Neural Networks',
    authors: [
      { name: 'Michael Y. Hu' },
      { name: 'Shreyans Jain', me: true },
      { name: 'Sangam Chaulagain' },
      { name: 'Naomi Saphra' },
    ],
    venue: 'ICLR 2025 Blogpost Track',
    year: 2025,
    selected: true,
    links: [
      {
        label: 'blogpost',
        href: 'https://iclr-blogposts.github.io/2025/blog/visualizing-training/',
      },
      {
        label: 'code',
        href: 'https://github.com/shreyansjainn/visualizing-training',
      },
    ],
  },
  {
    title: 'Sycophancy as Compositions of Atomic Psychometric Traits',
    authors: [
      { name: 'Shreyans Jain', me: true },
      { name: 'Alexandra Yost' },
      { name: 'Amirali Abdullah' },
    ],
    venue: 'BlackboxNLP 2025',
    year: 2025,
    status: 'Extended Abstract',
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2508.19316' }],
  },
  {
    title:
      'Towards Discovering Linguistic Indicators for Misalignment in Language Models',
    authors: [
      { name: 'Shreyans Jain', me: true },
      { name: 'Shivam Raval' },
    ],
    venue: 'BlackboxNLP 2025',
    year: 2025,
    status: 'Extended Abstract',
    links: [{ label: 'Zenodo', href: 'https://doi.org/10.5281/zenodo.16988484' }],
  },
];
