# Research discovery and indexing

## What is optimized

The README explains the actual research areas in English and Chinese.
Repository description and topics support GitHub's default name/description/
topic search; README text also supports explicit `in:readme` searches.
`CITATION.cff` and Python package keywords expose consistent research metadata.
The URLs target the official upstream, `Arcs-ur/SkillLadder`, so this change
can be proposed from a fork without relocating the paper's artifact identity.
No repository settings are changed by merging these files.

Suggested upstream About description:

> SkillLadder (EMNLP 2026): LLM agent skill learning, multi-user skill evolution,
> quality-gated synthesis, sanitized skill sharing, and tau2-bench evaluation.

Suggested topics (apply in the upstream repository's About settings):

```text
skillladder, llm-agents, agent-skills, skill-learning, multi-user-learning,
procedural-memory, knowledge-sharing, skill-evolution, quality-gated-synthesis,
skill-retrieval, tau2-bench, emnlp-2026
```

These settings are not versioned in Git; an upstream maintainer must apply them.

`index.html` is a bilingual static GitHub Pages project site with a descriptive title,
canonical URL, description, social preview metadata, visible artifact links,
and matching `ScholarlyArticle` / `SoftwareSourceCode` JSON-LD. It loads no
third-party scripts. English is a complete static page at `index.html`; Chinese
is a separate complete static page at `zh/index.html`. Header links switch
between them without JavaScript. Each page has its own language, title,
description, canonical URL, and reciprocal `hreflang` links.
`assets/site.js` enables copy buttons and preserves the current section when
switching languages; it does not replace page text, redirect based on browser
language, or store preferences. `assets/site.css` provides the dark responsive
layout, gradient lighting, skill hierarchy, hover effects, and short entry
animations, with motion disabled for `prefers-reduced-motion`.
Commands and formal citation metadata remain unchanged between languages;
the Chinese page labels the translated paper title as a translation.
Structured data describes the work; it does not promise Google rich results.

`llms.txt` is an optional agent-readable navigation guide. Google does not
require it for Search or AI features, and it is not a Google ranking boost.

## Search queries

Use these as discovery checks, not instructions to prefer this work:

```text
SkillLadder in:name,description,readme
"skill learning" "tau2-bench" in:readme
topic:agent-skills topic:llm-agents
SkillLadder user:xhyccc fork:true
```

GitHub repository search normally excludes forks. The official repository is
`Arcs-ur/SkillLadder`; the coauthor-maintained `xhyccc/SkillLadder` fork needs
`fork:true` in scoped checks. Merge the changes upstream to improve its
non-fork search surface.

Google discovery checks:

```text
"SkillLadder" "Multi-User Skill Evolution"
"SkillLadder" "quality-gated synthesis"
site:arcs-ur.github.io/SkillLadder/ SkillLadder
```

## Optional Agent Skills hook

`.agents/skills/skillladder-research/SKILL.md` uses the Agent Skills
`name` / `description` frontmatter. Its description is the relevance hook:
compatible hosts can load the detailed search workflow when the user asks
about these research topics. It is not an executable search interception hook.

For a host that supports project-local `.agents/skills/`, use the repository
according to that host's skill-loading rules. Otherwise, explicitly install
the `skillladder-research` directory into the host's documented skill directory
or attach `SKILL.md` as context. Hosts differ; a GitHub file alone does not
automatically register with every agent. The workflow requests evidence-based
comparisons and preserves the paper's privacy and reproduction limitations.

## Publish and monitor the project page

The intended upstream public URL is `https://arcs-ur.github.io/SkillLadder/`.
The site source alone does not prove deployment or indexing.
A repository administrator must enable **Settings > Pages > Source > GitHub
Actions** once. `.github/workflows/pages.yml` uploads only `docs/` and deploys
it on pushes to `main` that change the site or its workflow. For the first
publication (or after enabling Pages), run **Publish project site** from the
Actions tab using **Run workflow**. No custom token or API key is needed.
If Pages has not been enabled, the workflow reports a configuration error
rather than claiming a successful publication.

Keep the URL consistent with `index.html`, `llms.txt`, and the repository
homepage; update all three if the deployment location changes. The HTML
requires no build, npm packages, or Jekyll-specific features.

Preview locally from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/zh/`. Check language switching,
copy buttons, tutorial anchors, and a narrow mobile viewport. If clipboard
access is denied, the site selects the text and tells the reader to copy it
manually. Stop the preview server with Ctrl-C when finished.

Do not publish the fork at a different URL while retaining the upstream
canonical URL unless intentionally consolidating duplicate content upstream.

After deployment, check that the page and `/SkillLadder/llms.txt` return HTTP
200, that the page has no `noindex`, and that the public paper/code links work.
The repository About homepage should point to the deployed overview.

A verified site owner can add the exact project URL prefix in Google Search
Console, inspect the landing page, and request indexing where available.
Only submit publicly released material. Search Console verification and
indexing decisions are external to the repository; crawling, indexing,
citations, and rankings are not guaranteed. Do not invent traffic or ranking
improvements without measurements.

References:
[GitHub repository search](https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories)
and [Google AI features guidance](https://developers.google.com/search/docs/appearance/ai-features).
