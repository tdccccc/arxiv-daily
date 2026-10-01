import type { TopicSeed } from "./topics";

export interface TopicTemplate {
  id: string;
  name: string;
  category: string;
  /** Seeds: `topicFromSeed` turns each into a topic with its directions. */
  topics: TopicSeed[];
}

export const TOPIC_TEMPLATES: TopicTemplate[] = [
  {
    id: "blank",
    name: "Blank",
    category: "astro-ph",
    topics: [],
  },
  {
    id: "astro-ml",
    name: "Astrophysics + ML",
    category: "astro-ph",
    topics: [
      { name: "Photo-z",        tag: "photo-z",        directions: ["Photometric redshift methods, catalogs, comparisons."], detail: true },
      { name: "Galaxy Cluster", tag: "galaxy-cluster", directions: ["Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical."], detail: true },
      { name: "ML in Astro",    tag: "ml-astro",       directions: ["Deep learning, simulation-based inference, and related ML/DL applications in astrophysics."], detail: false },
    ],
  },
  {
    id: "nlp",
    name: "NLP / LLMs",
    category: "cs.CL",
    topics: [
      { name: "LLM Training", tag: "llm-training", directions: ["Pre-training, fine-tuning, RLHF, scaling laws, mixture-of-experts."], detail: true },
      { name: "RAG",          tag: "rag",          directions: ["Retrieval-augmented generation, vector stores, hybrid retrieval."], detail: true },
      { name: "Alignment",    tag: "alignment",    directions: ["Safety, interpretability, jailbreaks, constitutional AI."], detail: true },
      { name: "Evaluation",   tag: "eval",         directions: ["Benchmarks, leaderboards, evaluation methodology, contamination."], detail: false },
    ],
  },
  {
    id: "cv",
    name: "Computer Vision",
    category: "cs.CV",
    topics: [
      { name: "Diffusion", tag: "diffusion", directions: ["Diffusion-based image / video / 3D generation models."], detail: true },
      { name: "3D Vision", tag: "3d-vision", directions: ["NeRF, Gaussian splatting, 3D reconstruction, depth estimation."], detail: true },
      { name: "Video",     tag: "video",     directions: ["Video understanding, generation, action recognition."], detail: false },
    ],
  },
  {
    id: "bio",
    name: "Bioinformatics",
    category: "q-bio",
    topics: [
      { name: "Protein Structure", tag: "protein-structure", directions: ["Structure prediction, AlphaFold-style models, protein design."], detail: true },
      { name: "Genomics ML",       tag: "genomics-ml",       directions: ["Foundation models for genomics, single-cell, sequence modeling."], detail: true },
      { name: "Drug Discovery",    tag: "drug-discovery",    directions: ["Molecular generation, docking, binding affinity prediction."], detail: false },
    ],
  },
];
