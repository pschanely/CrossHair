# CrossHair

[![Build Status](https://travis-ci.org/pschanely/CrossHair.svg?branch=master)](https://travis-ci.org/pschanely/CrossHair) [![Join the chat at https://gitter.im/Cross_Hair/Lobby](https://badges.gitter.im/Cross_Hair/Lobby.svg)](https://gitter.im/Cross_Hair/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A static analysis tool for Python that blurs the line between type systems and testing.

If you have functions with [type annotations](https://www.python.org/dev/peps/pep-0484/) and add some invariants in the (defunct) [PEP 316](https://www.python.org/dev/peps/pep-0316/) syntax, CrossHair will attempt to find counterexamples for you:

![Animated GIF demonstrating the verification of a python function](doc/duplicate_list.gif)

CrossHair works by repeatedly calling your functions with fake symbolic values in an attempt to explore execution paths and find counterexamples.
This is not a new idea; it was first described in [this paper](https://hoheinzollern.files.wordpress.com/2008/04/seer1.pdf).
However, to my knowledge, CrossHair is the most complete implementation of the idea: it has at least some support for (possibly nested) lists, dicts, sets, and custom/mutable objects.

## Get Started

> **_NOTE:_**  CrossHair is in a highly experimental state right now. If you're using it, it's because you want it to succeed, want to help, are interested in the tech, or (hopefully) all of the above.

Inside the development environment of the code you want to analyze (virtual environment, conda environment, etc), install:
```shell
pip install git+https://github.com/pschanely/crosshair
```

CrossHair works best when it sits in its own window and thinks about your code while you work on it. Open such a window, activate your development environment, and run:
```shell
crosshair watch [directory with code to analyze]
```
You should then see perodically updating text that looks like "Analyzed 51 paths in 19 classes/functions." If CrossHair can find counterexamples for your postconditions, it'll tell you about them in this window.

Hit Ctrl-C to exit.

## IDE Integrations

Not yet, but stay tuned. As mentioned above, CrossHair wants to run in the background, but IDE integrations could reflect CrossHair results directly in your editor.

## Limitations

An (incomplete) list of present limitations. Some of these will be lifted over time (your help is welcome!), some may never be lifted.

* Automated theorem provers have very different perspectives on hard problems and easy problems than humans.
  * Be prepared to be surprised both by what CrossHair can tell you, and what it cannot.
* Only function and class definitions at the top level are anlyzed. (i.e. not when nested inside other functions/classes)
* Only deteministic behavior can be analyzed. (your code always does the same thing when starting with the same values)
  * In some cases, CorssHair can detect non-determinism and tell you about it.
* Symbolic values are largely implemented as Python proxies. CrossHair monkey-patches the system to maintain a good illusion, but the illusion is not complete:
  * Code that casres about the identity values (x is y) may not be fully analyzable.
  * Code that cares about the types of values may not be fully analyzable.

