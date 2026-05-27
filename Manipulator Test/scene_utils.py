
import os

def generate_dynamic_scene(obstacles, base_xml="scene_base_gripper.xml", out_xml="scene_dynamic.xml"):
    """
    Generates a MuJoCo XML file `out_xml` from `base_xml` 
    by injecting the given list of obstacles into <worldbody>.
    
    obstacles: List of dicts: e.g.
    [
        {'type': 'box', 'x': 0.5, 'y': 0.5, 'z': 0.5, 'dx': 0.1, 'dy': 0.1, 'dz': 0.1, 'color': 'Red'}
    ]
    """
    if not os.path.exists(base_xml):
        # Fallback to non-gripper if gripper base not found
        if base_xml == "scene_base_gripper.xml" and os.path.exists("scene_base.xml"):
             base_xml = "scene_base.xml"
        else:
             print(f"Error: Base XML {base_xml} not found.")
             return base_xml # Fallback
        
    try:
        with open(base_xml, "r") as f:
            content = f.read()
            
        injection = "\n    <!-- Dynamic Obstacles -->\n"
        
        for i, obs in enumerate(obstacles):
            t = obs.get('type', 'box').lower()
            x, y, z = obs.get('x', 0), obs.get('y', 0), obs.get('z', 0)
            c_name = obs.get('color', 'Gray')
            
            # Map Color Name to RGBA
            if c_name == 'Red': rgba = "0.8 0.2 0.2 1"
            elif c_name == 'Green': rgba = "0.2 0.8 0.2 1"
            elif c_name == 'Blue': rgba = "0.2 0.2 0.8 1"
            else: rgba = "0.5 0.5 0.5 1"
            
            if t == 'box':
                dx, dy, dz = obs.get('dx', 0.1), obs.get('dy', 0.1), obs.get('dz', 0.1)
                injection += f'    <geom name="d_obs_{i}" type="box" pos="{x} {y} {z}" size="{dx} {dy} {dz}" rgba="{rgba}"/>\n'
                
            elif t == 'sphere':
                r = obs.get('r', 0.1)
                injection += f'    <geom name="d_obs_{i}" type="sphere" pos="{x} {y} {z}" size="{r}" rgba="{rgba}"/>\n'
                
            elif t == 'cylinder':
                r = obs.get('r', 0.1)
                h = obs.get('h', 0.1)
                # MuJoCo cylinder size is radius, half-height
                injection += f'    <geom name="d_obs_{i}" type="cylinder" pos="{x} {y} {z}" size="{r} {h}" rgba="{rgba}"/>\n'
                
        injection += "    <!-- End Dynamic Obstacles -->\n"
        
        # Insert into <worldbody>
        # Simple string manipulation to find end of worldbody
        # We assume <worldbody> exists and ends with </worldbody>
        idx = content.rfind("</worldbody>")
        if idx == -1:
            print("Error: No </worldbody> tag found in base XML.")
            return base_xml
            
        new_content = content[:idx] + injection + content[idx:]
        
        with open(out_xml, "w") as f:
            f.write(new_content)
            
        return out_xml
        
    except Exception as e:
        print(f"Failed to generate scene: {e}")
        return base_xml
