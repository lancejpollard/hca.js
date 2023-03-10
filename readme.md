<br/>
<br/>
<p align='center'>
  <img src='https://github.com/lancejpollard/hca.js/blob/make/7-3.gif?raw=true' height='500'>
</p>
<br/>
<br/>
<p align='center'>
  <img src='https://github.com/lancejpollard/hca.js/blob/make/3-7.png?raw=true' height='200'>
  <img src='https://github.com/lancejpollard/hca.js/blob/make/5-4.png?raw=true' height='200'>
  <img src='https://github.com/lancejpollard/hca.js/blob/make/9-3.png?raw=true' height='200'>
</p>
<br/>

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

## Coordinate System in the Hyperbolic Plane

A major problem with hyperbolic tessellations is the idea of a
coordinate system, since it is infinite and intuitively thought of to
have a center. As the tiling rotates or moves, it changes the coordinate
system to adjust to a new ideal center. An introduction to this topic is
found in Volume 2 of professor Margenstern's book, in
[chapter 1 section 4](https://books.google.com/books?id=eEgvfic3A4kC&lpg=PP1&pg=PA70#v=onepage&q&f=false).

- [Coordinate systems for the hyperbolic plane](https://en.wikipedia.org/wiki/Coordinate_systems_for_the_hyperbolic_plane)

## Motions in the Hyperbolic Plane

There are 3 types of motion in the hyperbolic plane:

1. rotation
1. shift
1. glide

A glide is a combination of rotations and shifts. These are described in
volume 1 section 1.2.2.

## Spanning Trees of { p, q } Tilings

[This paper](https://arxiv.org/pdf/0911.4040.pdf) describes how spanning
trees work for { p, q } tilings when q is odd or even. The "even" case
is thoroughly described on page 422 of volume 1 of his books.

The tilings { p, q } are combinatoric and the language of the splittings
is regular when p >= 4. The language of the splittings is _not_ regular
for tilings { 3, q }, the case of equilateral triangles.

"Fibonacci trees" are described in
[volume 1 chapter 4](https://books.google.com/books?id=wGjX1PpFqjAC&lpg=PP1&pg=PA151#v=onepage&q&f=false).
They seem to be a special case of the pentagrid and heptagrid tilings.

## How Rendering Works

The Poincar?? disk model (and other general perspective projections) is
obtained from the hyperboloid model by projecting from the point ( 0, 0,
-1 ).

## Poincar?? Disk Model

The visualization of the tessellations uses the Poincar?? disc model,
which is described in volume 1 section 1.3.

## Models of Hyperbolic Geometry

[Here](http://roguetemple.com/z/hyper/models.php) is a somewhat complete
list of models of hyperbolic geometry.

- Poincar?? disk model
- Poincar?? half-plane model
- Upper half plane model
- [Minkowski hyperboloid model](https://en.wikipedia.org/wiki/Hyperboloid_model)
- [Klein-Beltrami disk model](https://en.wikipedia.org/wiki/Beltrami%E2%80%93Klein_model)
- [Gans disk model](https://en.wikipedia.org/wiki/Hyperbolic_geometry#The_Gans_model)
- Inverted Poincar?? disk model
- Hemisphere model
- [Band model](https://en.wikipedia.org/wiki/Band_model)
- Spiral model
- Polygonal model
- Joukowsky model
- Poincar?? Ball model

## General Notes

- [Uniform tilings in hyperbolic plane](https://en.wikipedia.org/wiki/Uniform_tilings_in_hyperbolic_plane)
- [Euclidean tilings by convex regular polygons](https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons)
- [Schwarz triangle](https://en.wikipedia.org/wiki/Schwarz_triangle)
- [List of uniform polyhedra by Schwarz triangle](https://en.wikipedia.org/wiki/List_of_uniform_polyhedra_by_Schwarz_triangle)
- http://people.hws.edu/mitchell/tilings/Part3.html
- [HyperRogue dev notes](http://www.roguetemple.com/z/hyper/dev.php)
- [HyperRogue docs](https://zenorogue.github.io/hyperrogue-doc/)
  - [hyperpoint.cpp](https://github.com/zenorogue/hyperrogue/blob/master/hyperpoint.cpp)
- [Illuminating hyperbolic geometry](https://www.youtube.com/watch?v=eGEQ_UuQtYs)
- https://github.com/HackerPoet/HyperEngine
- [The ??-Stereographic Model for Hyperbolic, Spherical and Euclidean Geometry](https://andbloch.github.io/K-Stereographic-Model)
- https://geoopt.readthedocs.io/en/latest/extended/stereographic.html
- [Tensorflow Docs](https://js.tensorflow.org/api/latest)
  - https://www.tensorflow.org/guide/tensor
- [Cellular Automata in Hyperbolic Plane simulator](https://dmishin.github.io/hyperbolic-ca-simulator/help.html)

## Knuth-Bendix Algorithm

> The knuth-bendix algorithm is a way of finding a set of rewrite rules
> for transforming words into a canonical form, where you can check if
> they're equal by just seeing if the words are the same. The reason
> this is relevant here is that the transformations of the hyperbolic
> plane (or whatever space) that leave the tiling fixed forms a group.
> If you imagine a certain tile being at the origin, these
> transformations will move different tiles onto the origin -- if two
> transformations are the same, they put the same tile onto the origin
> (potentially with a different rotation), so you can use that
> transformation to represent that tile (plus its rotation) and then if
> you ignore the rotation (e.g. by picking the smallest group element by
> some fixed ordering) then it really is a representation of the tile
> that you can compare for equality

From @panic on the
[HyperRogue discord](https://discord.com/channels/540092734300618753/694122059097178172).
