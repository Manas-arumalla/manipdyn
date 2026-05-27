
import mujoco
import numpy as np

class CollisionChecker:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def is_collision(self, q):
        """
        Check if configuration q is in collision.
        """
        # Save state
        # (Optimization: RRT is sequential, maybe no need to save/restore if we don't care about curr state)
        # But to be safe for integration validation:
        q_save = self.data.qpos[:6].copy()
        
        # Set candidate state
        self.data.qpos[:6] = q
        
        # Update geometry pipeline
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        
        # Check contacts
        # ncon is number of detected contacts
        # We need to filter out contacts that are "allowed" or "geometry <-> floor" if relevant?
        # Usually self-collisions are already defined in XML exclusions.
        # We assume ANY contact is bad for now, except maybe feet-floor (base is fixed).
        
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Get geom IDs
            g1 = contact.geom1
            g2 = contact.geom2
            
            # Use geom_bodyid to check if it's the robot vs Static
            # or Robot vs Robot (Self collision)
            # For simplicity, if dist < 0, it's a penetration
            if contact.dist < -0.001: # Small tolerance
                collision = True
                break
                
        # Restore state
        self.data.qpos[:6] = q_save
        # No need to re-run kinematics immediately unless needed
        
        return collision
