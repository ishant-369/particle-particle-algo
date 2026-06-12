import numpy as np
import matplotlib.pyplot as plt
N = 100
steps = 100
dt= 0.01
k=1.0
epsilon = 0.5
box_min = 0
box_max = 10
positions = np.random.rand(N, 2) * 10
velocities = np.zeros((N, 2))
charges = np.random.choice([-1, 1], N)
mass= np.ones(N)
plt.ion()
for step in range(steps):
    forces=np.zeros((N, 2))
    for i in range(N):
        for j in range(i+1, N):
            r = positions[j] - positions[i]
            distance = np.linalg.norm(r)
            if distance > 0.1:
                force_mag = (
                    k
                    * charges[i]
                    * charges[j]
                    / (distance**2 + epsilon**2)
                )

                force_vec = force_mag * r/ distance
                forces[i] += force_vec
                forces[j] -= force_vec
        
        #now code for updating the particles
        accelerations = forces/mass[:,None]
        velocities += accelerations * dt
        positions += velocities * dt
        #Left and right walls
        mask = positions[:, 0] < box_min
        positions[mask, 0] = box_min
        velocities[mask, 0] *= -1

        mask = positions[:, 0] > box_max
        positions[mask, 0] = box_max
        velocities[mask, 0] *= -1

        #Bottom and top walls
        mask = positions[:, 1] < box_min
        positions[mask, 1] = box_min
        velocities[mask, 1] *= -1

        mask = positions[:, 1] > box_max
        positions[mask, 1] = box_max
        velocities[mask, 1] *= -1
        
        if steps % 5 ==0:
            plt.clf()
            colors =[
                "red" if q>0 else "blue"
                for q in charges
            ]
            plt.scatter(positions[:,0], positions[:,1], c=colors)
            plt.xlim(0,10)
            plt.ylim(0,10)
            plt.title(f"Step {step}")
            plt.pause(0.01)
plt.ioff()
plt.show()