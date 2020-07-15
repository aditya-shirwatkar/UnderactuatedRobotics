" Modeling Dynamical Systems"
# This Tutorial provides a short tutorial for modeling input-output dynamical systems in Drake. It covers
## Writing your own simple dynamical systems,
## Simulating a dynamical system and plotting the results,
## Constructing a block diagram out of simple systems.

" Writing your own dynamics "
# In this section, we will describe how you can write your own dynamics 
# class in pydrake. Every user system should derive from the 
# pydrake.systems.framework.LeafSystem class.

# However, many dynamical systems we are interested in are 
# represented by a simple vector field in state-space form, 
# where we often use x to denote the state vector, u for the input vector, 
# and y for the output vector. 
# To make it even easier to implement systems in this form, 
# we have another subclass pydrake.systems.primitives.SymbolicVectorSystem 
# that makes it very easy to author these simple systems.
# u → System → y












