I'll be implementing few algorithms in Underactuated Robotics and also solving some interesting problems
- [Pendulum with Vibrating Base](#pendulum-with-vibrating-base)
- [Double Integrtor](#double-integrator)
  * [LQR (Linear Quadratic Regulator)](#lqr-linear-quadratic-regulator)

## Pendulum with Vibrating Base
<p>Consider <a href = https://github.com/aditya-shirwatkar/UnderactuatedRobotics/tree/master/Pendulum%20with%20Vibrating%20Base> here </a>
 an actuated pendulum whose base (pivot of the rod) is forced to oscillate horizontally according to the harmonic law
<img src="https://latex.codecogs.com/gif.latex?h&space;*&space;\sin{\omega&space;t}" title="h * \sin{\omega t}" />
 , where <i>h</i> denotes the amplitude of the oscillation and &omega; the frequency. The equation of motion for this system is</p>

<p><img src="https://latex.codecogs.com/gif.latex?m&space;l^2&space;\ddot&space;\theta&space;&plus;&space;m&space;g&space;l&space;\sin&space;\theta&space;=&space;m&space;l&space;\omega^2&space;h&space;\sin&space;(\omega&space;t)&space;\cos&space;\theta&space;&plus;&space;u." title="m l^2 \ddot \theta + m g l \sin \theta = m l \omega^2 h \sin (\omega t) \cos \theta + u." /></p>

<p>The goal is to design a time-dependent policy <img src="https://latex.codecogs.com/gif.latex?u&space;=&space;\pi&space;(\theta,&space;\dot&space;\theta,&space;t)" title="u = \pi (\theta, \dot \theta, t)" /> that makes the pendulum spin at constant velocity <img src="https://latex.codecogs.com/gif.latex?\dot&space;\theta&space;=&space;1" title="\dot \theta = 1" />,</p>
 
 
<p>below is the result through feedback cancellation,</p>

<p><img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Pendulum%20with%20Vibrating%20Base/vibPen_const_ang_vel.gif" /></p>


## Double Integrator
<p>Consider the double integrator system</p>

<p><img src="https://latex.codecogs.com/gif.latex?\ddot{q}&space;=&space;u,&space;\quad&space;|u|&space;\le&space;1." title="\ddot{q} = u, \quad |u| \le 1." /></p>

<p>If you would like a mechanical analog of the system, then you can think about this as a unit mass brick moving along the x-axis on a frictionless surface, with a control input which provides a horizontal force, <i>u</i> . The task is to find a policy <img src="https://latex.codecogs.com/gif.latex?\pi&space;({\bf{x}},&space;t)" title="\pi ({\bf{x}}, t)" /> for the system where, 
 <img src="https://latex.codecogs.com/gif.latex?{\bf{x}}=[q,\dot{q}]^T" title="{\bf{x}}=[q,\dot{q}]^T" /> to regulate this brick to <img src="https://latex.codecogs.com/gif.latex?{\bf{x}}=[0,0]^T" title="{\bf{x}}=[0,0]^T" /> 
.
</p>

#### LQR (Linear Quadratic Regulator)
<p><a href="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/lqr.ipynb">here</a></p>
<p><img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/d_i_lqr.gif" /></p>
