"""Shared fixtures.

Nothing here is mocked. `workdir` works because every path constant in src/ is a
*relative* Path, so it resolves against the current directory at call time --
chdir into a tmp dir and the real functions write there instead of data/.
"""

import subprocess
import textwrap

import pytest


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A throwaway project root. Real code writes here instead of data/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    return tmp_path


def long_markdown(title="Retrieval", sections=3, sentences=40):
    """Markdown long enough that the size splitter has to do real work."""
    body = f"# {title}\n\nIntro paragraph for {title}.\n"
    for i in range(sections):
        filler = " ".join(
            f"Sentence {j} about retrieval, embeddings and vector search."
            for j in range(sentences)
        )
        body += f"\n## Section {i}\n\n{filler}\n\n### Sub {i}\n\n{filler}\n"
    return body


@pytest.fixture
def html_page():
    """Real HTML, the shape trafilatura is meant to handle."""
    paragraphs = "".join(
        f"<h2>Section {i}</h2><p>{'A real sentence about retrieval systems. ' * 15}</p>"
        for i in range(3)
    )
    return textwrap.dedent(f"""\
        <html><head>
          <title>Building A Generative AI Platform</title>
          <meta name="author" content="Chip Huyen">
        </head><body><article>
          <h1>Building A Generative AI Platform</h1>
          {paragraphs}
        </article></body></html>""")


@pytest.fixture
def git_source(tmp_path_factory):
    """A real local git repo, cloneable by fetch_git. No network."""
    repo = tmp_path_factory.mktemp("origin")
    (repo / "guide").mkdir()
    (repo / "guide" / "interview.md").write_text(long_markdown("Interview"), "utf-8")
    (repo / "README.md").write_text("# Field Guide\n\nIndex page.\n", encoding="utf-8")
    (repo / "STYLING.md").write_text("# Styling\n\nExcluded.\n", encoding="utf-8")
    (repo / "notes.txt").write_text("not markdown", encoding="utf-8")

    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@t.t")
    run("config", "user.name", "t")
    run("add", "-A")
    run("commit", "-qm", "init")

    return {
        "id": "field-guide",
        "type": "git",
        "url": repo.as_uri(),
        "exclude": ["STYLING.md"],
    }
