SSC0250/Computer Graphics - Class assignments
---
###### How to run it


To run it, follow these steps: 
```
    python3 -m venv .
    source bin/activate
    pip install -r requirements.txt
    python3 <file.py>
```


These commands setup a virtual python3 environment at current directory and install its dependencies.  


###### About it


There were three assignments, named, respectively, __t1__, __t2__ and __t3__.


__t1__ consists in experimenting basic matrix transformations to develop a spring jumping to left or right.  
To ensure a physically coherent movimentation, we implemented __semi implicit euler method__ as integrator, to integrate basic equations of motion, using it with a semi-fixed timestep algorithm to define our __deltatime__, guided by [Glenn Fiedler's](https://gafferongames.com/) article series about game physics.  


__t2__ and __t3__ are very similar. They aim to implement a __MVP matrix__(a.k.a. __Model View Projection__) to render imported __.obj__ models in a 3D scene. __t3__ extends __t2__ functionalities to include local illumination, implementing __Phong reflection model__ to a slightly modified environment, with some model changes from previous assignment.


###### Authors 
- Ricardo A. Araujo - 936489  
- Tiago E. Triques - 9037713  
  
###### References 
- ["__Fix Your Timestep!__"](https://www.gafferongames.com/post/fix_your_timestep/), by Glenn Fiedler
