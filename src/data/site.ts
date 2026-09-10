// Site-wide config. Edit this one file to change identity, nav, and socials.

export const site = {
  name: 'Shreyans Jain',
  title: 'Shreyans Jain',
  tagline: 'Independent interpretability researcher',
  description:
    'Shreyans Jain — independent interpretability researcher working on how high-level behaviours in language models decompose into identifiable mechanisms, and how they form during training.',
  url: 'https://shreyansjainn.github.io',
  email: 'jshrey8@gmail.com',
  cvPdf:
    'https://drive.google.com/file/d/1Q0DZX5LjM6Ty36EPSn4wbvmFDQ5Okog1/view?usp=sharing',
};

// Anchor links scroll the single-page home; Blog is a real page.
export const nav = [
  { label: 'about', href: '/#about' },
  { label: 'research', href: '/#research' },
  { label: 'publications', href: '/#publications' },
  { label: 'writing', href: '/blog' },
  { label: 'cv', href: '/#cv' },
  { label: 'outside work', href: '/#outside' },
];

export const socials = [
  { label: 'Email', href: 'mailto:jshrey8@gmail.com', icon: 'email' },
  {
    label: 'Google Scholar',
    href: 'https://scholar.google.com/citations?user=kPbV2RYAAAAJ',
    icon: 'scholar',
  },
  { label: 'GitHub', href: 'https://github.com/shreyansjainn', icon: 'github' },
  {
    label: 'LinkedIn',
    href: 'https://www.linkedin.com/in/shreyans-jain-4b063667',
    icon: 'linkedin',
  },
  { label: 'X', href: 'https://x.com/py_parrot', icon: 'x' },
];
