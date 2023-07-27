import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def is_inside_triangle(x, y, x0, y0, x1, y1, x2, y2):
    dX0 = x - x0
    dY0 = y - y0
    dX1 = x - x1
    dY1 = y - y1
    dX2 = x - x2
    dY2 = y - y2

    cross0 = np.sign(dX0 * dY1 - dX1 * dY0)
    cross1 = np.sign(dX1 * dY2 - dX2 * dY1)
    cross2 = np.sign(dX2 * dY0 - dX0 * dY2)

    return (cross0 == cross1) & (cross1 == cross2) & (cross0 != 0)

def create_triangular_object(Nx, Ny):
    triangle_size = Ny // 4  # Adjust the size of the triangle
    x0, y0 = Nx // 4, Ny // 2 
    x1, y1 = Nx // 4 + triangle_size, Ny // 2 
    x2, y2 = Nx // 4, Ny // 2 - triangle_size
    X, Y = np.meshgrid(range(Nx), range(Ny))
    return is_inside_triangle(X, Y, x0, y0, x1, y1, x2, y2)
def main():
    """ Lattice Boltzmann Simulation """
    
    # Simulation parameters
    Nx = 400    # resolution x-dir
    Ny = 100    # resolution y-dir
    rho0 = 100    # average density
    tau = 0.6    # collision timescale
    Nt = 2000   # number of timesteps
    plotRealTime = True # switch on for plotting as the simulation goes along
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
    
    # Initial Conditions
    F = np.ones((Ny, Nx, NL))
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho
    
    # Triangular object boundary
    triangular_object = create_triangular_object(Nx, Ny)
    
    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)

    # def animate(it):
     #   plt.cla()
    
    # Simulation Main Loop
    for it in range(Nt):
        print(it)
        
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        
        # Set reflective boundaries for triangular object
        bndryF_triangular = F[triangular_object, :]
        bndryF_triangular = bndryF_triangular[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        
        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        
        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)
        
        F += -(1.0 / tau) * (F - Feq)
        
        # Apply boundary for triangular object
        F[triangular_object, :] = bndryF_triangular
        
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            ux[triangular_object] = 0
            uy[triangular_object] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[triangular_object] = np.nan
            vorticity = np.ma.array(vorticity, mask=triangular_object)
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(~triangular_object, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)    
            ax.set_aspect('equal')    
            plt.pause(0.001)
    
    # Create animation
   # ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=50, blit=False)

    # Save animation as video
  #  ani.save('latticeboltzmann.mp4', writer='ffmpeg', fps=15)        
    # Save figure
    plt.savefig('latticeboltzmann.png', dpi=240)
  # plt.show()

    return 0

if __name__ == "__main__":
    main()
