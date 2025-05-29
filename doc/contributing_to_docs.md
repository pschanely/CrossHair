# How to Contribute to Documentation

## How to Build

Change to the root directory of the repository.

Activate your virtual environment.

Install the requirements:

```
pip3 install -e .[dev]
```

Build with Sphinx:

```
cd doc
sphinx-build source build
```

The documentation is in the `doc/build` directory.

## Style Guide

**Headings.**
Use the following heading style from [Sphinx tutorial][sphinx-sections]:

[sphinx-sections]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections

* `#` with overline, for parts
* `*` with overline, for chapters
* `=`, for sections
* `-`, for subsections
* `^`, for subsubsections
* `“`, for paragraphs

**Line length.**
Use 88 characters limit.

Put sentences on individual lines to make diff'ing easier.


## Creating resources for an LLM

You might want to use something like NotebookLM to ask questions about CrossHair. The doc pages can be combined into a single pdf:
```sphinx-build source build -b pdf```

We can also dump github issues on the command line:
```
gh issue list -s all --limit 1000 --json title,state,url,createdAt,body,comments | python -c 'import json,sys; print("\n".join("=====\nTitle: {}\nState: {}\nURL: {}\n\nDate: {}\n\nBody:\n{}\n\n{}\n".format(i["title"], i["state"], i["url"], i["createdAt"], i["body"], "\n".join("Comment ({}, {}):\n{}\n".format(c["author"]["login"], c["createdAt"], c["body"]) for c in i["comments"])) for i in json.load(sys.stdin)))' > ./crosshair_issues.txt
```

Discussions are probably also possible, at `https://api.github.com/repos/pschanely/CrossHair/discussions` (and comments at e.g. `https://api.github.com/repos/pschanely/CrossHair/discussions/97/comments`), but I haven't attempted this yet.
