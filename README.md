<img src="doc/source/logo-gray.png" width="5%" align="left">

# CrossHair

[![Join the chat at https://gitter.im/Cross_Hair/Lobby](https://badges.gitter.im/Cross_Hair/Lobby.svg)](https://gitter.im/Cross_Hair/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Check status](https://github.com/pschanely/CrossHair/workflows/Check/badge.svg)](https://github.com/pschanely/CrossHair/actions?query=workflow%3ACheck)
[![Downloads](https://pepy.tech/badge/crosshair-tool)](https://pepy.tech/project/crosshair-tool)

An analysis tool for Python that blurs the line between testing and 
type systems.

> **_THE LATEST NEWS:_**  
Check out the new
[crosshair cover](https://crosshair.readthedocs.io/en/latest/cover.html)
command which finds inputs to get you code coverage.


If you have a function with
[type annotations](https://www.python.org/dev/peps/pep-0484/) and add a
contract
[in a supported syntax](https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html),
CrossHair will attempt to find counterexamples for you:

![Animated GIF demonstrating the verification of a python function](doc/source/duplicate_list.gif)

CrossHair works by repeatedly calling your functions with symbolic inputs.
It uses an [SMT solver] (a kind of theorem prover) to explore viable 
execution paths and find counterexamples for you.
This is not a new idea; a Python approach was first described in
[this paper].
However, to my knowledge, CrossHair is the most complete implementation:
it supports symbolic lists, dictionaries, sets, and custom classes.

[SMT solver]: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
[this paper]: https://hoheinzollern.files.wordpress.com/2008/04/seer1.pdf

Try CrossHair right now, in your browser, at [crosshair-web.org]!

CrossHair has [IDE integrations] for [VS Code], [PyCharm], and more.

[IDE integrations]: https://crosshair.readthedocs.io/en/latest/ide_integrations.html
[VS Code]: https://marketplace.visualstudio.com/items?itemName=mristin.crosshair-vscode
[PyCharm]: https://plugins.jetbrains.com/plugin/16266-crosshair-pycharm

[crosshair-web.org]: https://crosshair-web.org

Want to do me a favor? Sign up for 
[email](http://eepurl.com/hGTLRH)
or [RSS](https://pschanely.github.io/feed.xml)
updates.
There are
[other ways to help](https://crosshair.readthedocs.io/en/latest/how_can_i_help.html)
too.

## Documentation
 
See https://crosshair.readthedocs.io/
