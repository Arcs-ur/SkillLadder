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

`index.html` is a static GitHub Pages project overview with a descriptive title,
canonical URL, description, social preview metadata, visible artifact links,
and matching `ScholarlyArticle` / `SoftwareSourceCode` JSON-LD. It loads no
third-party scripts and does not require JavaScript to read or follow links.
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
This PR prepares the page; it does not deploy it or prove it is indexed.
After merging, publish GitHub Pages from the `main` branch's `/docs` directory in repository
Settings > Pages. Keep this URL consistent with `index.html`, `llms.txt`, and
the repository homepage; update all three if the deployment location changes.
The HTML can be served directly without Jekyll-specific features.

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
