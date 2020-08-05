SSC0250/ Computer Graphics - Class assignments
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


T2 and T3 are very similar. They aim to implement MVP matrix(a.k.a. Model View Projection) to render imported _.obj__ models in a 3D scene. T3 extends T2 functionalities to include local illumination, implementing Phong reflection model to a slightly modified environment, with some model changes from previous assignment.


###### Authors 
- Ricardo A. Araujo - 936489  
- Tiago E. Triques - 9037713  
  
###### References 
- ["__Fix Your Timestep!__"](https://www.gafferongames.com/post/fix_your_timestep/), by Glenn Fiedler
