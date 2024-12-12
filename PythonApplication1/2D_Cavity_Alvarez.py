import numpy as np
from matplotlib import pyplot as plt, cm

plt.rcParams['interactive'] == True


nx = 41
x_max = 2 
x_min = 0

ny = 41
y_max = 2
y_min = 0

nt = 500
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, x_max, ny)
X, Y = np.meshgrid(x,y)

rho = 1 # density
nu = .1 # viscosity coefficient
dt = .001 # timestep

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

def set_velocity_boundary(u,v):
    # Velocidty boundary conditions (Dirichlet or "fixed")
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1    # set velocity on cavity lid equal to 1
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

    return u, v

def set_pressure_boundary(p):
    # Pressure boundary conditions (Neumann or "second-type")
    '''
    Trick question: Would the approach below break depending on 
    whether you use forward, backward, or centered differences?
    Why or why not?
    '''
    p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
    p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
    p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
    # Pressure boundary conditions (Dirichlet or "fixed")
    p[-1, :] = 0        # p = 0 at y = 2

    return p

def diff_1st_x(f,dx):
  return (f[1:-1, 2:] - f[1:-1,0:-2])/(2*dx)

def diff_1st_y(f,dy):
  return (f[2:, 1:-1] - f[0:-2,1:-1])/(2*dy)

def diff_2nd_x(f,dx):
  return (
      # f(i-1,j) + 2*f(i,j) + f(i+1,j) 
      (f[1:-1,0:-2] - 2*f[1:-1,1:-1] + f[1:-1,2:])
      /dx**2
      ) 
  
def diff_2nd_y(f,dy):
  return (
      # f(i,j-1) + 2*f(i,j) + f(i,j+1) 
      (f[0:-2,1:-1] - 2*f[1:-1,1:-1] + f[2:,1:-1])
      /dy**2
      )

def laplacian(f, dx, dy):
  return (diff_2nd_x(f, dx) + 
          diff_2nd_y(f, dy))

def compute_vel_star(un, vn, dx, dy, dt, nu):
  u_star = un.copy()
  v_star = vn.copy()

  # Equation (7), Owkes (2020)
  u_star[1:-1,1:-1] = (nu * laplacian(un, dx, dy) -
                         (un[1:-1, 1:-1]*diff_1st_x(un,dx) + 
                          vn[1:-1, 1:-1]*diff_1st_y(un,dy))
                      )*dt + un[1:-1, 1:-1]
  '''
  Pre-class challenge:
  * Can you rewrite this function 
  to avoid the repetitive lines of code?
  * If so, do your changes sacrifice 
  efficiency for readibility? Why or why not?
  '''

  # Equation (12), Owkes (2020)
  v_star[1:-1,1:-1] = (nu * laplacian(vn, dx, dy) -
                         (un[1:-1, 1:-1]*diff_1st_x(vn,dx) + 
                          vn[1:-1, 1:-1]*diff_1st_y(vn,dy))
                      )*dt + vn[1:-1, 1:-1]
  
  return u_star, v_star

def get_b(u_star, v_star, dx, dy, dt):
  '''
  Right hand side of equation (6), Owkes (2020)
  '''
  # b comes from starred velocities, which have boundary
  # conditions (the starred velocity is just the velocity
  # without the influence of pressure, so it still maintains
  # the boundary conditions of the velocity),  
  # but b does not really have boundary conditions of its own.
  # (P.S. Important point is how the value is used in pressure_poisson function,
  # i.e., `b[1:-1, 1:-1]` or simply `b`)
  # b = np.zeros((ny, nx))

  # (Divergence of vector-valued function)
  divergence_vel_star = (diff_1st_x(u_star,dx) + 
                         diff_1st_y(v_star,dy))
  # b[1:-1,1:-1] = (1*rho/dt) * divergence_vel_star
  b = (rho/dt) * divergence_vel_star

  return b

def pressure_poisson(p, b, dx, dy):  
  pn = p.copy()

  # (pn(i-1,j) + pn(i+1,j)) * dy**2 
  term1 = (pn[1:-1,0:-2] + pn[1:-1,2:]) * dy**2
  # (pn(i,j-1) + pn(i,j+1)) * dx**2
  term2 = (pn[0:-2,1:-1] + pn[2:,1:-1]) * dx**2
  # term 3 is -b * dx^2  * dy^2
  term3 = -(b * (dx**2 * dy**2))

  p[1:-1,1:-1] = (
      (term1 + term2 + term3)
      / (2*(dx**2+dy**2))
  )
  
  # reset boundary condition
  p = set_pressure_boundary(p)
  
  return p

def corrector_step(u_star, v_star, p, dx, dy, dt, rho):
  u = np.empty_like(u_star)
  v = np.empty_like(u_star)
  
  u[1:-1,1:-1] = (-dt/rho)*(diff_1st_x(p, dx)) + u_star[1:-1,1:-1]
  v[1:-1,1:-1] = (-dt/rho)*(diff_1st_y(p, dy)) + v_star[1:-1,1:-1]

  # reset velocity boundary
  u, v = set_velocity_boundary(u, v)

  return u, v

def simulate_cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
  for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Predictor step (starred velocities)
    u_star, v_star = compute_vel_star(un, vn, dx, dy, dt, nu)

    # Solve pressure Poisson Equation
    b = get_b(u_star, v_star, dx, dy, dt)
    p = pressure_poisson(p, b, dx, dy)

    # Corrector step (update velocities)
    u, v = corrector_step(u_star, v_star, p, dx, dy, dt, rho)

  return u, v, p

u, v, p = simulate_cavity_flow(1000, u, v, dt, dx, dy, p, rho, nu)

fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)
plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
