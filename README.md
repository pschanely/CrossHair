<img src="doc/source/logo.png" width="5%" align="left">

# CrossHair

[![Join the chat at https://gitter.im/Cross_Hair/Lobby](https://badges.gitter.im/Cross_Hair/Lobby.svg)](https://gitter.im/Cross_Hair/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![codecov](https://codecov.io/gh/pschanely/CrossHair/branch/master/graph/badge.svg)](https://codecov.io/gh/pschanely/CrossHair)
[![Check status](https://github.com/pschanely/CrossHair/workflows/Check/badge.svg)](https://github.com/pschanely/CrossHair/actions?query=workflow%3ACheck)

An analysis tool for Python that blurs the line between testing and 
type systems.

> **_THE LATEST NEWS:_**  
[Marko Ristin] just published **several** new [IDE integrations] for CrossHair,
including plugins for [VS Code] and [PyCharm].

[Marko Ristin]: https://github.com/mristin
[IDE integrations]: https://crosshair.readthedocs.io/en/latest/ide_integrations.html
[VS Code]: https://marketplace.visualstudio.com/items?itemName=mristin.crosshair-vscode
[PyCharm]: https://plugins.jetbrains.com/plugin/16266-crosshair-pycharm


If you have a function with
[type annotations](https://www.python.org/dev/peps/pep-0484/) and add a
contract
[in a supported syntax](https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html),
CrossHair will attempt to find counterexamples for you:

![Animated GIF demonstrating the verification of a python function](doc/source/duplicate_list.gif)

CrossHair works by repeatedly calling your functions with symbolic inputs.
It uses an [SMT solver] (a kind of theorem prover) to explore viable 
execution paths and find counterexamples for you.
This is not a new idea; an approach for Python was first described in
[this paper].
However, to my knowledge, CrossHair is the most complete implementation of 
the idea: it supports symbolic lists, dictionaries, sets, and even symbolic
instances of your own classes.

[SMT solver]: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
[this paper]: https://hoheinzollern.files.wordpress.com/2008/04/seer1.pdf

Try CrossHair right now, in your browser, at [crosshair-web.org]!

[crosshair-web.org]: https://crosshair-web.org

> **_NOTE:_**  CrossHair is in an experimental state right now.
> [You can help though!]

[You can help though!]: https://crosshair.readthedocs.io/en/latest/how_can_i_help.html

## Documentation
 
Available at https://crosshair.readthedocs.io/
