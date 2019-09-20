# CrossHair

[![Join the chat at https://gitter.im/Cross_Hair/Lobby](https://badges.gitter.im/Cross_Hair/Lobby.svg)](https://gitter.im/Cross_Hair/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A static analysis tool for Python that blurs the line between types and testing.

![Animated GIF demonstrating the verification of a python function](doc/duplicate_list.gif)

## Limitations

An (incomplete) list of present limitations. Some of these will be lifted over time (your help is welcome!), some may never be lifted.

* Automated theorem provers have very different perspectives on hard problems and easy problems than humans.
  * Be prepared to be surprised both by what CrossHair can tell you, and what it cannot.
* Nested function and class definitions are not anlyzed.
* Only deteministic behavior can be analyzed. (your code always does the same thing when starting with the same values)
  * In some cases, CorssHair can detect non-determinism and tell you about it.
* Symbolic values are largely implemented as Python proxies. CrossHair monkey-patches the system to maintain a good illusion, but the illusion is not complete:
  * Code that casres about the identity values (x is y) may not be fully analyzable.
  * Code that cares about the types of values may not be fully analyzable.

