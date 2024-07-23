## Black Hole Dynamics


The fundamental entities in our simulation are Big Atoms and potential
energy functions. Big Atoms have fundamental properties like mass, charge,
position, internal energy, and so on. The potential energy functions
describe the interactions or forces between Big Atoms.

An important aspect of the simulation is the time evolution of the system
under the influence of the forces. The time evolution is governed by the
equations of motion, which are derived from the potential energy functions.

A particularly important potential energy function is the gravitational
potential energy function. The gravitational potential energy function
describes the interaction between two Big Atoms due to gravity.

The potential energy function for the gravitational interaction between two
Big Atoms is given by:

$$
U(\vec{r}) = -\frac{G m_1 m_2}{|r|}
$$

where:
- $G$ is the gravitational constant,
- $m_1$ and $m_2$ are the masses of the two Big Atoms, and
- $\vec{r}$ is the displacement vector between the two Big Atoms and $|\vec{r}|$ is the magnitude of the displacement vector.

The graviational force between two Big Atoms is given by the negative
gradient of the potential energy function:
$$
\vec{F} = -\nabla U(\vec{r}) = -\frac{G m_1 m_2}{r^3} \vec{r}
$$

### Schwarzschild Radius

The Schwarzschild radius is a characteristic radius associated with a black
hole. It is defined as the radius at which the escape velocity equals the
speed of light. The Schwarzschild radius of a black hole of mass $M$ is given
by:

$$
r_s = \frac{2 G M}{c^2}
$$

where:
- $G$ is the gravitational constant,
- $M$ is the mass of the black hole, and
- $c$ is the speed of light.

The Schwarzschild radius defines the event horizon of a black hole, beyond
which nothing can escape, not even light.

Technically, if we imagine a Big Atom as a point particle, then
every Big Atom has a positive Schwarzschild radius, although repulsion
forces would likely prevent the formation of a black hole in "normal"
circumstances. However, even if one formed, the small Schwarzschild radius
would be a small target for other Big Atoms to "hit", so in practice no
Big Atom would be able to get within the Schwarzschild radius of another
Big Atom. Besides, we disable gravitational interactions between Big Atoms
that are separated by a distance less than some minimum threshold, for
numerical stability reasons.

Of course, if the Big Atom had a sufficiently large mass and its radius
was reasonably large and less than the Schwarzschild radius, then its event
horizon would extend beyond its radius and likely beyond any radius of
strong repulsion forces. In this case, it would be fair to characterize
the Big Atom as a black hole.

### Black Hole Dynamics of Big Atoms

It is easy to check if a single Big Atom is a black hole.

1. Compute the Schwarzschild radius of the Big Atom.
2. Check if the Big Atom's radius is at least a sizable fraction of the
   Schwarzschild radius (Big Atoms often do overlap, so it is not necessary
   for the Big Atom to be smaller than the Schwarzschild radius).

This is kind of an ad hoc way to determine if a Big Atom is a black hole. It
is interesting in a sense, and due to Hawking radiation, smaller black holes
very quickly evaporate, so it may be the case that such small black holes
do form in our real universe. However, they definitely can form in our
simulation under the right conditions, and of course they can be easily
constructed by design.

### Black Hole Dynamics of Clusters of Big Atoms

If you arrange a cluster of Big Atoms in a way that the Schwarzschild radius
of the cluster is larger than the radius of the cluster, then the cluster
can be considered a black hole.

We already have a way of clustering Big Atoms into composites. While this
approach may fail to identify many regions as black holes, it is a good
starting point.

We can quickly compute bounding spheres for composites and check if the
Schwarzschild radius of the composite is larger than the radius of the
composite. If so, we can consider the composite a black hole.

This is likely something that can happen in our simulation through the
"natural" evolution of the system. It is also something that can be
constructed by design.

When a composite is identified as a black hole, we can choose to merge the
Big Atoms in the composite into a single Big Atom with the mass of the
composite and the position of the center of mass of the composite. We can
also conserve all of the relevant properties of the Big Atoms, such as
momentum, angular momentum, mass, charge, and magnetic moment.

If the simulation runs long enough, it is likely that black holes will form
and eventually, it will enter a "Black Hole Era" where most of the Big Atoms
are part of black holes. Of course this is only a theoretical insight, because
in practice the simulation will not run long enough to reach this point, although
you could tweak the parameters to make it happen more quickly, but it may fail
to generate interesting dynamics in the process.










