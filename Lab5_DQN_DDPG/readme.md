# Lab5: Deep Q-Network and Deep Deterministic Policy Gradient
Implement two deep reinforcement algorithms by completing the following three tasks:
(1) Solve LunarLander-v2 using deep Q-network (DQN)  
(2) Solve LunarLanderContinuous-v2 using deep deterministic policy gradient (DDPG)   
(3) Solve BreakoutNoFrameskip-v4 using deep Q-network (DQN)  

## Game Environment – LunarLander-v2
Introduction: Rocket trajectory optimization is a classic topic in optimal control. Coordinates are the first two numbers in state vector, where landing pad is always at coordinates (0,0). Reward for moving from the top of the screen to landing pad and zero speed is about 100 to 140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

* Observation  
1. Horizontal Coordinate
2. Vertical Coordinate
3. Horizontal Speed
4. Vertical Speed
5. Angle
6. Angle Speed
7. If first leg has contact
8. If second leg has contact

* Four discrete actions available: Action [4]: 0 (No-op), 1 (Fire left engine), 2 (Fire main engine), 3 (Fire right engine)

![LunarLander-v2](https://github.com/frankkn/NYCU_DLP/blob/master/Lab5_DQN_DDPG/images/LunarLander-v2.gif)

## Game Environment – LunarLanderContinuous-v2
* Introduction: same as LunarLander-v2.
* Observation [8]: same as LunarLander-v2.
* Action [2]:
    * Main engine: -1 to 0 off, 0 to +1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    * Left-right: -1.0 to -0.5 fire left engine, +0.5 to +1.0 fire right engine, -0.5 to 0.5 off

![BreakoutNoFrameskip-v4](https://github.com/frankkn/NYCU_DLP/blob/master/Lab5_DQN_DDPG/images/BreakoutNoFrameskip-v4.gif)

## Game Environment – BreakoutNoFrameskip-v4
Introduction: Another famous Atari game. The dynamics are similar to pong: You move a paddle and hit the ball in a brick wall at the top of the screen. Your goal is to destroy the brick wall. You can try to break through the wall and let the ball wreak havoc on the other side, all on its own! You have five lives. Detailed documentation can be found on the AtariAge page.

* Observation: By default, the environment returns the RGB image that is displayed to human players as an observation.
* Action [4]: 0 (No-op), 1 (Fire), 2 (Right), 3 (Left)
