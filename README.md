I'll be implementing few algorithms in Underactuated Robotics and also solving some interesting problems

## Table of Contents
- [List of Custom Environments](#list-of-custom-environments)
- [Quadrotor-2D](#quadrotor-2d)
  * [Trajectory Optimization on Quadrotor-2D](#trajectory-optimization-on-quadrotor-2d)
  * [LQR on Quadrotor-2D](#lqr-on-quadrotor-2d)
- [CartPole Continuous](#cartpole-continuous)
  * [LQR on CartPole Contiunous](#lqr-on-cartpole-continuous)
- [Double Integrator](#double-integrator)
  * [Value Iteration on Double Integrator](#value-iteration-on-double-integrator)
  * [LQR on Double Integrator](#lqr-on-double-integrator)
  * [Trajectory Optimization on Double Integrator](#trajectory-optimization-on-double-integrator)
- [Pendulum with Vibrating Base](#pendulum-with-vibrating-base)


## List of Custom Environments
The below is a list of custom environments that I built using <a href="https://gym.openai.com/">OpenAI gym</a>, feel free to use them however you want
  * [Quadrotor-2D](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/custom-environments/gym_custom_envs/envs/quadrotor_2d_env.py)
  * [Horizontally Vibrating Pendulum](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/custom-environments/gym_custom_envs/envs/vibrating_pendulum_env.py)
  * [Double Integrator](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/custom-environments/gym_custom_envs/envs/double_integrator_env.py)

## Quadrotor-2D
<p> Consider <a href="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/custom-environments/gym_custom_envs/envs/quadrotor_2d_env.py">here</a> an extremely simple model of a quadrotor that is restricted to live in the plane. The equations of motion are almost trivial, since it is only a single rigid body, and certainly fit into our standard manipulator equations: </p>

<p><img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/gifs/quadrotor-2d.png" width="120" height="107"/> </p>
<p>
<img src="https://latex.codecogs.com/gif.latex?\begin{}&space;\\m&space;\ddot{x}&space;=&space;-(u_1&space;&plus;&space;u_2)\sin\theta,&space;\label{eq:quad_x}\\&space;m&space;\ddot{y}&space;=&space;(u_1&space;&plus;&space;u_2)\cos\theta&space;-&space;mg,&space;\label{eq:quad_y}\\&space;I&space;\ddot\theta&space;=&space;r&space;(u_1&space;-&space;u_2)&space;\label{eq:quad_theta}&space;\end{}" title="\begin{} \\m \ddot{x} = -(u_1 + u_2)\sin\theta, \label{eq:quad_x}\\ m \ddot{y} = (u_1 + u_2)\cos\theta - mg, \label{eq:quad_y}\\ I \ddot\theta = r (u_1 - u_2) \label{eq:quad_theta} \end{}" width="150" height="60"/>
</p>

### Trajectory Optimization on Quadrotor-2D
[here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/trajectory_optimisation.ipynb)

![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/gifs/trajopt_quad2d.gif)
![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/gifs/trajopt_quadrotor2d.png)

### LQR on Quadrotor-2D
LQR [here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/lqr.ipynb) works essentially out of the box for Quadrotors, if linearized around a nominal fixed point (where the non-zero thrust from the propellers is balancing gravity).
In this case nominal fixed point is 
<p>
 <img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;x^*\\&space;y^*\\&space;\theta&space;^*\\&space;\dot{x}&space;^*\\&space;\dot{y}&space;^*\\&space;\dot{\theta}&space;^*&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;0\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0&space;\end{bmatrix}&space;and&space;\begin{bmatrix}&space;u_1^*\\&space;u_2^*&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\frac{mg}{2}\\&space;\frac{mg}{2}&space;\end{bmatrix}" title="\begin{bmatrix} x^*\\ y^*\\ \theta ^*\\ \dot{x} ^*\\ \dot{y} ^*\\ \dot{\theta} ^* \end{bmatrix} = \begin{bmatrix} 0\\ 0\\ 0\\ 0\\ 0\\ 0 \end{bmatrix} and \begin{bmatrix} u_1^*\\ u_2^* \end{bmatrix} = \begin{bmatrix} \frac{mg}{2}\\ \frac{mg}{2} \end{bmatrix}" width="160" height="107"/>
</p>

<img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Quadrotor%202D%20/gifs/quadrotor_lqr.gif" width="400" height="250"/>

## CartPole Continuous

The [here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/custom-environments/gym_custom_envs/envs/cart_pole_conti_env.py) task is to balance a simple pendulum around its unstable equilibrium, using only horizontal forces on the cart. Balancing the cart-pole system is used in many introductory courses in control because it can be accomplished with simple linear control (e.g. pole placement) techniques.
Consider the full swing-up and balance control problem, which requires a full nonlinear control treatment.

### LQR on CartPole Continuous
[here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Cart%20Pole%20Continuous/lqr.ipynb)

![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Cart%20Pole%20Continuous/gifs/cart_lqr.gif)

## Double Integrator
<p>Consider the double integrator system</p>

<p><img src="https://latex.codecogs.com/gif.latex?\ddot{q}&space;=&space;u,&space;\quad&space;|u|&space;\le&space;1." title="\ddot{q} = u, \quad |u| \le 1." /></p>

<p>If you would like a mechanical analog of the system, then you can think about this as a unit mass brick moving along the x-axis on a frictionless surface, with a control input which provides a horizontal force, <i>u</i> . The task is to find a policy <img src="https://latex.codecogs.com/gif.latex?\pi&space;({\bf{x}},&space;t)" title="\pi ({\bf{x}}, t)" /> for the system where, 
 <img src="https://latex.codecogs.com/gif.latex?{\bf{x}}=[q,\dot{q}]^T" title="{\bf{x}}=[q,\dot{q}]^T" /> to regulate this brick to <img src="https://latex.codecogs.com/gif.latex?{\bf{x}}=[0,0]^T" title="{\bf{x}}=[0,0]^T" /> 
.
</p>

### Value Iteration on Double Integrator
[here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/value_iteration.py)

![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/gifs/min_time_value_iter.gif)
![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/gifs/quad_value_iter.gif)

### LQR on Double Integrator

<p><a href="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/lqr.ipynb">here</a></p>
<p><img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/gifs/d_i_lqr.gif" /></p>

### Trajectory Optimization on Double Integrator
[here](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/trajectory_optimization.ipynb)

![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/gifs/d_i_trajopt.gif)
![](https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Double%20Integrator/gifs/trajopt_d_i.png)

## Pendulum with Vibrating Base

<p>Consider <a href = https://github.com/aditya-shirwatkar/UnderactuatedRobotics/tree/master/Pendulum%20with%20Vibrating%20Base> here </a>
 an actuated pendulum whose base (pivot of the rod) is forced to oscillate horizontally according to the harmonic law
<img src="https://latex.codecogs.com/gif.latex?h&space;*&space;\sin{\omega&space;t}" title="h * \sin{\omega t}" />
 , where <i>h</i> denotes the amplitude of the oscillation and &omega; the frequency. The equation of motion for this system is</p>

<p><img src="https://latex.codecogs.com/gif.latex?m&space;l^2&space;\ddot&space;\theta&space;&plus;&space;m&space;g&space;l&space;\sin&space;\theta&space;=&space;m&space;l&space;\omega^2&space;h&space;\sin&space;(\omega&space;t)&space;\cos&space;\theta&space;&plus;&space;u." title="m l^2 \ddot \theta + m g l \sin \theta = m l \omega^2 h \sin (\omega t) \cos \theta + u." /></p>

<p>The goal is to design a time-dependent policy <img src="https://latex.codecogs.com/gif.latex?u&space;=&space;\pi&space;(\theta,&space;\dot&space;\theta,&space;t)" title="u = \pi (\theta, \dot \theta, t)" /> that makes the pendulum spin at constant velocity <img src="https://latex.codecogs.com/gif.latex?\dot&space;\theta&space;=&space;1" title="\dot \theta = 1" />,</p>
 
 
<p>below is the result through feedback cancellation,</p>

<p><img src="https://github.com/aditya-shirwatkar/UnderactuatedRobotics/blob/master/Pendulum%20with%20Vibrating%20Base/gifs/vibPen_const_ang_vel.gif" /></p>
