import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
from pygame.locals import *
from Box2D import *
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody) 
from Box2D import b2Vec2
from circ_env.envs.line import Line
from circ_env.envs.dimensions import Dimensions

PPM = 100.0  # pixels per m
TARGET_FPS = 50
TIME_STEP = 1.0 / TARGET_FPS
WORLD_WIDTH, WORLD_HEIGHT = 450/PPM, 800/PPM

class ContactListener(b2ContactListener):
    def __init__(self):
        super().__init__()
        self._contact = False

    def BeginContact(self, contact):
        self._contact = True

    def EndContact(self, contact):
        self._contact = False

    def contact(self):
        return self._contact

class CircleEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": TARGET_FPS}

    def __init__(self, render_mode=None):
        super(CircleEnvironment, self).__init__()
        # Set the environment parameters
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT
        self.PPM = PPM
        self.current_step = 0
        # Initialize the Pybox2D world
        self.world = b2World(gravity=(0, 0))        

        ############ TUKAJ SPREMINJATE

        agent_radius_px = 30
        self.max_agent_vel = 2.0
        
        self.time_steps = 500

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                            high=np.array([self.width, self.height, self.width, self.height], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_agent_vel, high=self.max_agent_vel, shape=(2,), dtype=np.float32)

        self.create_agent(agent_radius_px) 
        object_radius_px = 30
        self.create_puck(object_radius_px, 'k')

        ############ DO TUKAJ SPREMINJATE
        # Set the contact listener
        self.contact_listener = ContactListener()
        self.world.contactListener = self.contact_listener

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None     

    def _get_obs(self):
      
        agent_pos = self.get_agent_position()
        pak_pos = self.get_puck_position()
       
        return np.concatenate((agent_pos, pak_pos))

    
    def reset(self, seed=None, options=None):
        self.world.ClearForces()
        # Reset the current step counter
        self.current_step = 0

        ############ TUKAJ SPREMINJATE

        self.reset_agent((np.random.uniform(self.agent_radius*1.5, self.width - self.agent_radius*1.5), np.random.uniform(self.agent_radius*1.5, self.height - self.agent_radius*1.5)))
        self.reset_puck((np.random.uniform(self.object_radius+self.agent_radius*3, self.width - self.object_radius-self.agent_radius*3), np.random.uniform(self.object_radius+self.agent_radius*3, self.height - self.object_radius-self.agent_radius*3)))
     
        ############ DO TUKAJ SPREMINJATE

        # Return the initial observation
        return self._get_obs(), {}
    
    def step(self, action):

        ############ TUKAJ SPREMINJATE
        #action = self.move_agent_mouse()
        self.set_agent_velocity(action) 

        obs = self._get_obs() 

        #self.object.ApplyAngularImpulse(-0.25*self.object.inertia*self.object.angularVelocity, True)
        #self.object.ApplyForce(-1*self.object.linearVelocity, self.object.worldCenter, True)        

        K = 2000
        B = 0.5
        m = 1

        pos_a = self.get_agent_position()
        pos_p = self.get_puck_position()
        dist = (self.agent_radius + self.object_radius) - self.calc_distance(pos_a, pos_p)

        FK = np.zeros_like(pos_p)
        if dist > 0:
            smer = self.unit_vector(pos_a, pos_p)
            FK = K * dist * smer

        vel_p = self.get_puck_velocity()
        FB = -B * vel_p

        F = np.zeros_like(pos_p)
        acc = np.zeros_like(pos_p)
        F = FK + FB
        acc = F / m
        self.set_puck_velocity(vel_p + acc * TIME_STEP)

        reward = -1.0/self.time_steps
        done = False          

        if self._is_collision(self.agent, self.object):
            reward += 0.05
            #done = True

        if self.is_puck_outside_screen():
            reward = 1.0
            done = True

        if self.current_step >= self.time_steps:
            done = True

        ############ DO TUKAJ SPREMINJATE
        self.current_step += 1
        self.world.Step(TIME_STEP, 6, 2)
        self.world.ClearForces()

        return obs, reward, done, done, {}       

    def render(self, render_mode=None):
        if self.render_mode == "human":
            return self._render_frame()


    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width*self.PPM, self.height*self.PPM))
            pygame.display.set_caption("Circle Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Define colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        yellow = (219,195,0)
        
        self.screen.fill(white)

        ############ TUKAJ SPREMINJATE

        self.draw_agent(green)
        self.draw_puck(yellow)

        ############ DO TUKAJ SPREMINJATE
        if self.render_mode == "human":
            # Update the Pygame display
            pygame.display.flip()
            pygame.display.update()
            pygame.event.pump()
            pygame.event.clear()
            self.clock.tick(self.metadata["render_fps"])

        
    def close(self):
        self.world.DestroyBody(self.agent)
        self.world.contactListener = None
        self.world = None
        
        # Quit Pygame
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _is_collision(self, object, goal):
        for fixture_object in object.fixtures:
            for fixture_goal in goal.fixtures:
                if b2TestOverlap(fixture_object.shape, 0, fixture_goal.shape, 0, object.transform, goal.transform):
                    return True
        return False   

    def create_agent(self, radius_px):
        self.agent_radius = radius_px/self.PPM
        friction = 0.0
        agent_fixture = b2FixtureDef(shape=b2CircleShape(radius=self.agent_radius), density=1.0, friction=friction, restitution=0.0)
        self.agent = self.world.CreateDynamicBody(position=(np.random.uniform(self.agent_radius*2, self.width - self.agent_radius*2), np.random.uniform(self.agent_radius*2, self.height - self.agent_radius*2)), fixtures=agent_fixture)

    def create_puck(self, radius_px, type):
        self.object_radius = radius_px/self.PPM
        friction = 0.0
        shape = b2CircleShape(radius=self.object_radius)
        object_fixture = b2FixtureDef(shape=shape, density=0.5, friction=friction, restitution=0.5)
        if (type == 'd'):
            self.object = self.world.CreateDynamicBody(position=(np.random.uniform(self.object_radius+self.agent_radius*3, self.width - self.object_radius-self.agent_radius*3), np.random.uniform(self.object_radius+self.agent_radius*3, self.height - self.object_radius-self.agent_radius*3)), fixtures=object_fixture)
        if (type == 'k'):
            self.object = self.world.CreateKinematicBody(position=(np.random.uniform(self.object_radius+self.agent_radius*3, self.width - self.object_radius-self.agent_radius*3), np.random.uniform(self.object_radius+self.agent_radius*3, self.height - self.object_radius-self.agent_radius*3)), fixtures=object_fixture)

    def create_goal(self, dim_px, position):
        goal_width, goal_height = dim_px[0]/self.PPM, dim_px[1]/self.PPM
        self.goal_fixture = b2FixtureDef(shape=b2PolygonShape(box=(goal_width/2, goal_height/2)), density=0.0, friction=0.5, restitution=0, isSensor=True)
        self.goal = self.world.CreateStaticBody(position=position, fixtures=self.goal_fixture)

    def create_circ_target(self, radius_px):
        self.target_radius = radius_px/self.PPM
        self.target = b2Vec2(float(np.random.uniform(self.target_radius*2, self.width - self.target_radius*2)), float(np.random.uniform(self.target_radius*2, self.height - self.target_radius*2)))
        
    def create_random_object(self, radius_px):
        # Generate random width and height for bounding box
        self.object_radius = radius_px/self.PPM
        points = []
        for i in range(4):
            angle = 2*math.pi*i/4 + np.random.uniform(-math.pi*i/4 / 2, math.pi*i/4 / 2)
            radius = self.object_radius + np.random.uniform(-self.object_radius/4, self.object_radius/2)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(b2Vec2(x, y))

        object_vertices = [b2Vec2(vec[0], vec[1]) for vec in points]

        self.object_density = 1.0
        self.object_friction = 1.0
        self.object_restitution = 0.0

        object_fixture = b2FixtureDef(shape=b2PolygonShape(vertices=object_vertices),
                           density=self.object_density, friction=self.object_friction, restitution=self.object_restitution)

        self.object = self.world.CreateDynamicBody(position=(np.random.uniform(self.object_radius+self.agent_radius*3, self.width - self.object_radius-self.agent_radius*3), np.random.uniform(self.object_radius+self.agent_radius*3, self.height - self.object_radius-self.agent_radius*3)), fixtures=object_fixture)

    def create_border(self):
        self.border = self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(0.0, 0.0), (self.width, 0.0)]),
                b2EdgeShape(vertices=[(0.0, self.height), (self.width, self.height)]),
                b2EdgeShape(vertices=[(0.0, 0.0), (0.0, self.height)]),
                b2EdgeShape(vertices=[(self.width, 0.0), (self.width, self.height)])
            ]
        ) 
        
    def get_agent_position(self):
        return np.array([self.agent.position.x, self.agent.position.y], dtype=np.float32)
    
    def get_agent_velocity(self):
        return np.array([self.agent.linearVelocity.x, self.agent.linearVelocity.y], dtype=np.float32)    
    
    def get_puck_position(self):
        return np.array([self.object.position.x, self.object.position.y], dtype=np.float32)
    
    def get_puck_velocity(self):
        return np.array([self.object.linearVelocity.x, self.object.linearVelocity.y], dtype=np.float32)
    
    def get_target_position(self):
        return np.array([self.target.x, self.target.y], dtype=np.float32)
    
    def reset_agent(self, position):
        self.agent.position = position
        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0.0

    def reset_puck(self, position):
        self.object.position = position
        self.object.linearVelocity = (0, 0)
        self.object.angularVelocity = 0.0

    def reset_target(self, position):
        self.target.x = position[0]
        self.target.y = position[1]

    def reset_random_object(self, position):
        self.object.position = position
        self.object.linearVelocity = (0, 0)
        self.object.angularVelocity = 0.0

        points = []
        base_radius = self.object_radius*np.random.uniform(0.2, 1.2)
        for i in range(4):
            angle = 2*math.pi*i/4 + np.random.uniform(-math.pi*i/4 / 2, math.pi*i/4 / 2)
            radius = base_radius + np.random.uniform(-base_radius/2, base_radius/2)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(b2Vec2(x, y))

        object_vertices = [b2Vec2(vec[0], vec[1]) for vec in points]      
        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=object_vertices),
            density=self.object_density,
            friction=self.object_friction,
            restitution=self.object_restitution
        )
        self.object.DestroyFixture(self.object.fixtures[0])
        self.object.CreateFixture(fixture_def)          

    def set_agent_velocity(self, vel):
        self.agent.linearVelocity = b2Vec2(float(vel[0]), float(vel[1]))
        self.agent.angularVelocity = 0.0       
 
    def set_puck_velocity(self, vel):
        self.object.linearVelocity = b2Vec2(float(vel[0]), float(vel[1]))
        self.object.angularVelocity = 0.0

    def limit_puck_velocity(self, maximal_velocity):
        # Check if the magnitude of velocity is larger than maximal_velocity
        velocity = self.object.linearVelocity
        if velocity.length > maximal_velocity:
            # If it is, calculate the unit vector of velocity and multiply it by maximal_velocity
            unit_velocity = velocity / velocity.length
            limited_velocity = unit_velocity * maximal_velocity
        else:
            # If it's not, return the original velocity
            limited_velocity = velocity
        self.object.linearVelocity = limited_velocity
        self.object.angularVelocity = 0.0        

    def draw_agent(self, color):
        pygame.draw.circle(self.screen, color, (int(self.agent.position.x*self.PPM), int(self.agent.position.y*self.PPM)), int(self.agent_radius*self.PPM))

    def draw_puck(self, color):
        pygame.draw.circle(self.screen, color, (int(self.object.position.x*self.PPM), int(self.object.position.y*self.PPM)), int(self.object_radius*self.PPM))

    def draw_border(self, color):
        pygame.draw.rect(self.screen, color, (1, 1, int(self.width*self.PPM-1), int(self.height*self.PPM-1)), 1)

    def draw_target(self, color):
        pygame.draw.circle(self.screen, color, (int(self.target.x*self.PPM), int(self.target.y*self.PPM)), int(self.target_radius*self.PPM))

    def draw_goal(self, color):
        goal_position = self.goal.position
        goal_dimensions = self.goal_fixture.shape.vertices
        goal_vertices = [(goal_position[0]*self.PPM + vertex[0]*self.PPM, goal_position[1]*self.PPM + vertex[1]*self.PPM) for vertex in goal_dimensions]
        pygame.draw.polygon(self.screen, color, goal_vertices, 0)

    def draw_random_object(self, color):
        # Convert Box2D vectors to Pygame points
        pos = self.object.position
        angle = -self.object.angle
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]], dtype=np.float32)        

        shape = self.object.fixtures[0].shape
        
        # Convert the shape's vertices to Pygame points
        vertices = [b2Vec2(v) for v in np.dot(shape.vertices, rotation)]
        vertices = [(v + pos) for v in vertices]
        points = [(int(v.x*self.PPM), int(v.y*self.PPM)) for v in vertices]      
        
        # Draw the polygon using Pygame
        pygame.draw.polygon(self.screen, color, points)                 
               
    def is_agent_outside_screen(self):

        outside = False

        if (self.agent.position.x < self.agent_radius) or (self.agent.position.x > (self.width - self.agent_radius)):     
            outside = True

        if (self.agent.position.y < self.agent_radius) or (self.agent.position.y > (self.height - self.agent_radius)):
            outside = True
        
        return outside

    def is_puck_outside_screen(self):

        outside = False

        if (self.object.position.x < self.object_radius) or (self.object.position.x > (self.width - self.object_radius)):     
            outside = True

        if (self.object.position.y < self.object_radius) or (self.object.position.y > (self.height - self.object_radius)):
            outside = True

        return outside

    def calc_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)
        
    def unit_vector(self, pos1, pos2):
        direction = pos2 - pos1
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            unit_direction = direction / magnitude
        else:
            unit_direction = np.zeros_like(direction)
        return unit_direction

    def calculate_component(self, pos_agent, pos_target, vel):
        # Calculate the unit vector pointing from pos1 to pos2
        direction = pos_target - pos_agent
        unit_vector = direction / np.linalg.norm(direction)

        # Calculate the component of vel in the direction of the unit vector
        component = np.dot(vel, unit_vector)

        # Return the scaled component
        return component
    
    def move_agent_mouse(self):
        x, y = 0, 0

        if pygame.mouse.get_pressed()[0]:
            try:
                mouse_pos = pygame.mouse.get_pos()
                agent_pos = self.get_agent_position()
                dx = agent_pos[0] - mouse_pos[0]/self.PPM
                dy = agent_pos[1] - mouse_pos[1]/self.PPM
                x, y = -dx*2, -dy*2
            except AttributeError:
                pass
 
        action = np.array([x,y],dtype=np.float32)

        return action   
