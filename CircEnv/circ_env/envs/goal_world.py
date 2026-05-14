import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from Box2D import *
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)
from Box2D import b2Vec2

PPM = 100.0
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


class GoalEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": TARGET_FPS}

    def __init__(self, render_mode=None):
        super(GoalEnvironment, self).__init__()
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT
        self.PPM = PPM
        self.current_step = 0
        self.world = b2World(gravity=(0, 0))

        agent_radius_px = 30
        object_radius_px = 30
        self.max_agent_vel = 2.0
        self.max_puck_vel = 5.0

        self.time_steps = 500

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.width, self.height, self.width, self.height], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-self.max_agent_vel, high=self.max_agent_vel, shape=(2,), dtype=np.float32)

        self.create_agent(agent_radius_px)
        self.create_puck(object_radius_px, 'd')
        self.create_goal((400, 10), (self.width/2, 10/self.PPM))
        self.create_border()

        self.contact_listener = ContactListener()
        self.world.contactListener = self.contact_listener

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _get_obs(self):
        agent_pos = self.get_agent_position()
        puck_pos = self.get_puck_position()
        return np.concatenate((agent_pos, puck_pos))

    def reset(self, seed=None, options=None):
        self.world.ClearForces()
        self.current_step = 0

        # Agent starts in lower half, puck in middle — helps with exploration
        self.reset_agent((
            np.random.uniform(self.agent_radius*1.5, self.width - self.agent_radius*1.5),
            np.random.uniform(self.height*0.5, self.height - self.agent_radius*1.5)
        ))
        self.reset_puck((
            np.random.uniform(self.object_radius + self.agent_radius*3, self.width - self.object_radius - self.agent_radius*3),
            np.random.uniform(self.height*0.25, self.height*0.75)
        ))

        return self._get_obs(), {}

    def step(self, action):
        self.set_agent_velocity(action)
        self.limit_puck_velocity(self.max_puck_vel)

        obs = self._get_obs()

        reward = -1.0/self.time_steps
        done = False

        if self.current_step >= self.time_steps:
            done = True

        if self._is_collision(self.object, self.goal):
            reward += 1.0
            done = True

        if self._is_collision(self.agent, self.object):
            goal_pos = np.array([self.goal.position.x, self.goal.position.y])
            coll_normal = self.get_puck_position() - self.get_agent_position()
            coll_normal = coll_normal / np.linalg.norm(coll_normal)
            F_comp = self.calculate_component(self.get_puck_position(), goal_pos, coll_normal)
            reward += 0.05 * F_comp

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
            self.screen = pygame.display.set_mode((int(self.width*self.PPM), int(self.height*self.PPM)))
            pygame.display.set_caption("Goal Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        white = (255, 255, 255)
        black = (0, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        yellow = (219, 195, 0)

        self.screen.fill(white)
        self.draw_goal(blue)
        self.draw_border(black)
        self.draw_agent(green)
        self.draw_puck(yellow)

        if self.render_mode == "human":
            pygame.display.flip()
            pygame.display.update()
            pygame.event.pump()
            pygame.event.clear()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        self.world.DestroyBody(self.agent)
        self.world.contactListener = None
        self.world = None
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
        agent_fixture = b2FixtureDef(shape=b2CircleShape(radius=self.agent_radius), density=1.0, friction=0.0, restitution=0.0)
        self.agent = self.world.CreateDynamicBody(
            position=(np.random.uniform(self.agent_radius*2, self.width - self.agent_radius*2),
                      np.random.uniform(self.agent_radius*2, self.height - self.agent_radius*2)),
            fixtures=agent_fixture
        )

    def create_puck(self, radius_px, type):
        self.object_radius = radius_px/self.PPM
        shape = b2CircleShape(radius=self.object_radius)
        object_fixture = b2FixtureDef(shape=shape, density=0.5, friction=0.0, restitution=0.5)
        pos = (np.random.uniform(self.object_radius + self.agent_radius*3, self.width - self.object_radius - self.agent_radius*3),
               np.random.uniform(self.object_radius + self.agent_radius*3, self.height - self.object_radius - self.agent_radius*3))
        if type == 'd':
            self.object = self.world.CreateDynamicBody(position=pos, fixtures=object_fixture)
        if type == 'k':
            self.object = self.world.CreateKinematicBody(position=pos, fixtures=object_fixture)

    def create_goal(self, dim_px, position):
        goal_width, goal_height = dim_px[0]/self.PPM, dim_px[1]/self.PPM
        self.goal_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(goal_width/2, goal_height/2)),
            density=0.0, friction=0.5, restitution=0, isSensor=True
        )
        self.goal = self.world.CreateStaticBody(position=position, fixtures=self.goal_fixture)

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

    def reset_agent(self, position):
        self.agent.position = position
        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0.0

    def reset_puck(self, position):
        self.object.position = position
        self.object.linearVelocity = (0, 0)
        self.object.angularVelocity = 0.0

    def set_agent_velocity(self, vel):
        self.agent.linearVelocity = b2Vec2(float(vel[0]), float(vel[1]))
        self.agent.angularVelocity = 0.0

    def set_puck_velocity(self, vel):
        self.object.linearVelocity = b2Vec2(float(vel[0]), float(vel[1]))
        self.object.angularVelocity = 0.0

    def limit_puck_velocity(self, maximal_velocity):
        velocity = self.object.linearVelocity
        if velocity.length > maximal_velocity:
            unit_velocity = velocity / velocity.length
            limited_velocity = unit_velocity * maximal_velocity
        else:
            limited_velocity = velocity
        self.object.linearVelocity = limited_velocity
        self.object.angularVelocity = 0.0

    def draw_agent(self, color):
        pygame.draw.circle(self.screen, color,
                           (int(self.agent.position.x*self.PPM), int(self.agent.position.y*self.PPM)),
                           int(self.agent_radius*self.PPM))

    def draw_puck(self, color):
        pygame.draw.circle(self.screen, color,
                           (int(self.object.position.x*self.PPM), int(self.object.position.y*self.PPM)),
                           int(self.object_radius*self.PPM))

    def draw_border(self, color):
        pygame.draw.rect(self.screen, color, (1, 1, int(self.width*self.PPM-1), int(self.height*self.PPM-1)), 1)

    def draw_goal(self, color):
        goal_position = self.goal.position
        goal_dimensions = self.goal_fixture.shape.vertices
        goal_vertices = [
            (goal_position[0]*self.PPM + v[0]*self.PPM, goal_position[1]*self.PPM + v[1]*self.PPM)
            for v in goal_dimensions
        ]
        pygame.draw.polygon(self.screen, color, goal_vertices, 0)

    def calc_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def unit_vector(self, pos1, pos2):
        direction = pos2 - pos1
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            return direction / magnitude
        return np.zeros_like(direction)

    def calculate_component(self, pos_agent, pos_target, vel):
        direction = pos_target - pos_agent
        unit = direction / np.linalg.norm(direction)
        return np.dot(vel, unit)
