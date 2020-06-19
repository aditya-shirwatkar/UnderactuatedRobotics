I'll be implementing few algorithms in Underactuated Robotics and also solving some interesting problems
- [Pendulum with Vibrating Base](#pendulum-with-vibrating-base)
## Pendulum with Vibrating Base
Consider an actuated pendulum whose base (pivot of the rod) is forced to oscillate horizontally according to the harmonic law
<img src="https://latex.codecogs.com/gif.latex?h&space;*&space;\sin{\omega&space;t}" title="h * \sin{\omega t}" />
 , where *h* denotes the amplitude of the oscillation and &omega; the frequency. The equation of motion for this system is
<p align="center"><img src="https://latex.codecogs.com/gif.latex?m&space;l^2&space;\ddot&space;\theta&space;&plus;&space;m&space;g&space;l&space;\sin&space;\theta&space;=&space;m&space;l&space;\omega^2&space;h&space;\sin&space;(\omega&space;t)&space;\cos&space;\theta&space;&plus;&space;u." title="m l^2 \ddot \theta + m g l \sin \theta = m l \omega^2 h \sin (\omega t) \cos \theta + u." /></p>
goal is to design a time-dependent control law <img src="https://latex.codecogs.com/gif.latex?u&space;=&space;\pi&space;(\theta,&space;\dot&space;\theta,&space;t)" title="u = \pi (\theta, \dot \theta, t)" /> that makes the pendulum spin at constant velocity <img src="https://latex.codecogs.com/gif.latex?\dot&space;\theta&space;=&space;1" title="\dot \theta = 1" />
