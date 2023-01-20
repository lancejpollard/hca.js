# hca.js

Hyperbolic cellular automata in the 2D plane in JavaScript. There is
also the sister library
[hyperbolic tessellations in the 2D plane](https://github.com/lancejpollard/ht.js),
which just renders static 2d hyperbolic tessellations. As opposed to
this hca.js library, which aims to implement much of what is found in
[Maurice Margenstern](http://www.lita.univ-lorraine.fr/~margens/)'s
books,
[Cellular Automata in Hyperbolic Spaces](https://www.amazon.com/Cellular-Automata-Hyperbolic-Spaces-Implementations/dp/1933153067)
volumes 1 and 2.

<br/>
<br/>
<p align='center'>
  <img src='https://github.com/lancejpollard/hca.js/blob/make/7.png?raw=true' height='500'>
</p>
<br/>

## Overview

This project aims to implement what is found in
[Maurice Margenstern](http://www.lita.univ-lorraine.fr/~margens/)'s
books,
[Cellular Automata in Hyperbolic Spaces](https://www.amazon.com/Cellular-Automata-Hyperbolic-Spaces-Implementations/dp/1933153067)
volumes 1 and 2. In there he describes extremely elegant properties,
proofs, and algorithms relating to cellular automata in the hyperbolic
plane. The goal with this project is to provide a simple tool to
generate these hyperbolic tessellations, which can be rendered,
animated, and interactive, in customizable yet intuitive ways.

This project will take many years before it becomes usable. If done
right, it will be a general model containing the coordinates of
hyperbolic tiling tiles, which can then be rendered into an interaction
component.

## Coordinate System

A major problem with hyperbolic tessellations is the idea of a
coordinate system, since it is infinite and intuitively thought of to
have a center. As the tiling rotates or moves, it changes the coordinate
system to adjust to a new ideal center. An introduction to this topic is
found in Volume 2 of professor Margenstern's book, in chapter 1
section 4.
