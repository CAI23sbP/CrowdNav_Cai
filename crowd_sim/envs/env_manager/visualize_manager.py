import numpy as np
from omegaconf import OmegaConf
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from typing import List, Optional, Union
import threading, time

class VisualizeManager():
    def __init__(
        self, 
        config: OmegaConf, 
        ):
        self.robot_fov = np.pi * config.env_config.robot.FOV
        self.human_fov = np.pi * config.env_config.humans.FOV
        self.max_range = config.obs_config.scan.max_range
        self.collision_cnt = 0
        self.time_limit = config.env_config.sim.time_limit
        self.window_size = config.visualize_config.window_size
        self.scale = config.visualize_config.scale
        self.viewer = None
        self.kinematics = config.env_config.robot.kinematics

    def get_scanned_humans(self,robot,human):
        rpx, rpy = robot.px, robot.py
        if type(human) is np.ndarray:
            dist = ((human[2] - rpx) ** 2 + (human[3] - rpy) ** 2) ** (1 / 2) 
            
        else:
            dist = ((human.px - rpx) ** 2 + (human.py - rpy) ** 2) ** (1 / 2) 

        if dist <= self.max_range:
            return True 
        else: 
            return False 

    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
        if state1.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        if type(state2) == np.ndarray:
            v_12 = [state2[2] - state1.px, state2[3] - state1.py]
        else:
            v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2
        
        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            dist = self.get_scanned_humans(state1,state2)
            if dist: 
                return True
            else:
                return False 
        else:
            return False

    def render(self,
                robot: Union[Robot, list],
                obstacle_vertices:List,
                lidar_angles,
                which_visible : List[bool],
                lidar_scan,
                sub_goal: Union[np.ndarray, None],
                humans: Union[List[Human], np.ndarray],
                path: Optional[np.ndarray]= None,
                sim_time: Optional[float] = None,
                lidar_scan_override=None,
                goal_override=None,
                return_rgb_array : bool = False,
                phase :str = 'train'
                ):

        if phase == 'test' :
            time.sleep(0.01)
        
        elif phase =='train':
            time.sleep(0.05)

        elif phase =='asking':
            time.sleep(0.01)
            
        WINDOW_W, WINDOW_H = self.window_size, self.window_size 
        VP_W = WINDOW_W
        VP_H = WINDOW_H
        from crowd_sim.envs.env_manager import rendering_manager as rendering
        import pyglet
        from pyglet import gl
        # Create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()
            self.transform.set_scale(self.scale, self.scale)
            self.transform.set_translation(int(WINDOW_W/2), int(WINDOW_H/2))
            if sim_time is not None:
                self.time_label = pyglet.text.Label(
                    '0000', font_size=12,
                    x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                    color=(255,255,255,255))
            self.currently_rendering_iteration = 0
            self.image_lock = threading.Lock()

        def make_circle(c, r, res=10):
            thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
            verts = np.zeros((res, 2))
            verts[:,0] = c[0] + r * np.cos(thetas)
            verts[:,1] = c[1] + r * np.sin(thetas)
            return verts

        # Render in pyglet
        with self.image_lock:
            self.currently_rendering_iteration += 1
            self.viewer.draw_circle(r=10, color=(0.3,0.3,0.3))
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()
            win.clear()
            gl.glViewport(0, 0, VP_W, VP_H)
            # colors
            bgcolor = np.array([0.1, 0.8, 0.4])
            obstcolor = np.array([0.3, 0.3, 0.3])
            goalcolor = np.array([1., 1., 0.3])
            goallinecolor =  np.array([1., 0., 1.])
            nosecolor = np.array([0.3, 0.3, 0.3])
            lidarcolor = np.array([1., 0., 0.])
            agentcolor = np.array([0., 1., 1.])
            unvisible_agentcolor = np.array([1., 0., 1.])
            # Green background
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
            gl.glVertex3f(0, VP_H, 0)
            gl.glVertex3f(VP_W, VP_H, 0)
            gl.glVertex3f(VP_W, 0, 0)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            self.transform.enable()
            if len(humans) > 0:
                if isinstance(humans[0], Human):
                    for index ,human in enumerate(humans):
                        px = human.px
                        py = human.py
                        angle = np.arctan2(human.vy, human.vx)
                        r = human.radius
                        poly = make_circle((px, py), r)
                        gl.glBegin(gl.GL_POLYGON)

                        if which_visible[index]:
                            gl.glColor4f(agentcolor[0], agentcolor[1], agentcolor[2], 1)
                        else:
                            gl.glColor4f(unvisible_agentcolor[0], unvisible_agentcolor[1], unvisible_agentcolor[2], 1)
                        for vert in poly:
                            gl.glVertex3f(vert[0], vert[1], 0)
                        gl.glEnd()
                        # Direction triangle
                        xnose = px + r * np.cos(angle)
                        ynose = py + r * np.sin(angle)
                        xright = px + 0.3 * r * -np.sin(angle)
                        yright = py + 0.3 * r * np.cos(angle)
                        xleft = px - 0.3 * r * -np.sin(angle)
                        yleft = py - 0.3 * r * np.cos(angle)
                        gl.glBegin(gl.GL_TRIANGLES)
                        gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                        gl.glVertex3f(xnose, ynose, 0)
                        gl.glVertex3f(xright, yright, 0)
                        gl.glVertex3f(xleft, yleft, 0)
                        gl.glEnd()
                else:
                    for index, human in enumerate(humans):
                        px, py = human[0], human[1]
                        vx, vy = human[2], human[3]
                        r = human[4]

                        angle = np.arctan2(vy, vx)
                        poly = make_circle((px, py), r)
                        gl.glBegin(gl.GL_POLYGON)
                        if which_visible[index]:
                            gl.glColor4f(agentcolor[0], agentcolor[1], agentcolor[2], 1)
                        else:
                            gl.glColor4f(unvisible_agentcolor[0], unvisible_agentcolor[1], unvisible_agentcolor[2], 1)
                        for vert in poly:
                            gl.glVertex3f(vert[0], vert[1], 0)
                        gl.glEnd()
                        # Direction triangle
                        xnose = px + r * np.cos(angle)
                        ynose = py + r * np.sin(angle)
                        xright = px + 0.3 * r * -np.sin(angle)
                        yright = py + 0.3 * r * np.cos(angle)
                        xleft = px - 0.3 * r * -np.sin(angle)
                        yleft = py - 0.3 * r * np.cos(angle)
                        gl.glBegin(gl.GL_TRIANGLES)
                        gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                        gl.glVertex3f(xnose, ynose, 0)
                        gl.glVertex3f(xright, yright, 0)
                        gl.glVertex3f(xleft, yleft, 0)
                        gl.glEnd()

            if isinstance(robot , Robot):
                px = robot.px
                py = robot.py
                if self.kinematics == 'holonomic':
                    angle = np.arctan2(robot.vy,robot.vx)
                else:
                    angle = robot.theta
                r = robot.radius
                xgoal = robot.gx
                ygoal = robot.gy
            else:
                px = robot[0]
                py = robot[1]
                if self.kinematics == 'holonomic':
                    angle = np.arctan2(robot[3],robot[2])
                else:
                    angle = robot[-1]
                r = robot[4]
                xgoal = robot[5]
                ygoal = robot[6]

            for poly in obstacle_vertices:
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
            # LIDAR rays
            scan = lidar_scan_override
            if scan is None:
                scan = lidar_scan
            lidar_angles = lidar_angles
            x_ray_ends = px + scan * np.cos(lidar_angles)
            y_ray_ends = py + scan * np.sin(lidar_angles)
            for ray_idx in range(len(scan)):
                end_x = x_ray_ends[ray_idx]
                end_y = y_ray_ends[ray_idx]
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                gl.glVertex3f(px, py, 0)
                gl.glVertex3f(end_x, end_y, 0)
                gl.glEnd()
            
            # Agent as Circle
            poly = make_circle((px, py), r)
            gl.glBegin(gl.GL_POLYGON)
            color = np.array([1., 1., 1.])
            gl.glColor4f(color[0], color[1], color[2], 1)
            for vert in poly:
                gl.glVertex3f(vert[0], vert[1], 0)
            gl.glEnd()
            # Direction triangle
            xnose = px + r * np.cos(angle)
            ynose = py + r * np.sin(angle)
            xright = px + 0.3 * r * -np.sin(angle)
            yright = py + 0.3 * r * np.cos(angle)
            xleft = px - 0.3 * r * -np.sin(angle)
            yleft = py - 0.3 * r * np.cos(angle)
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
            gl.glVertex3f(xnose, ynose, 0)
            gl.glVertex3f(xright, yright, 0)
            gl.glVertex3f(xleft, yleft, 0)
            gl.glEnd()

            # Goal
            if goal_override is not None:
                xgoal, ygoal = goal_override
            # Goal markers
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
            triangle = make_circle((xgoal, ygoal), r, res=3)
            for vert in triangle:
                gl.glVertex3f(vert[0], vert[1], 0)
            gl.glEnd()
            # Goal line
            if sub_goal is not None:
                subgoalcolor = np.array([1., 0., 1.])
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(subgoalcolor[0], subgoalcolor[1], subgoalcolor[2], 1)
                triangle = make_circle((sub_goal[0], sub_goal[1]), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()

            if path is not None:
                gl.glBegin(gl.GL_LINE_STRIP)
                for point in path:
                    gl.glVertex3f(point[0], point[1], 0)
                gl.glColor4f(goallinecolor[0], goallinecolor[1], goallinecolor[2], 1)
                gl.glEnd()

            # --
            self.transform.disable()
            # Text
            if sim_time is not None:
                self.time_label.text = ""
                self.time_label.text = "Time {} / Limit {}".format(round(sim_time,2), self.time_limit)
                self.time_label.draw()
            if not return_rgb_array:
                win.flip()
            arr = None
            if return_rgb_array:
                buffer = pyglet.image.get_buffer_manager().get_color_buffer()
                image_data = buffer.get_image_data()
                arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
                arr = arr.reshape(buffer.height, buffer.width, 4)
                arr = arr[::-1,:,0:3]
            return arr if return_rgb_array else self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.window.clear()
            self.viewer.close()