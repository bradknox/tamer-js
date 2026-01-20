# TAMER Web

A JavaScript port of the TAMER (Training an Agent Manually via Evaluative Reinforcement) framework, enabling interactive agent training directly in modern web browsers.

## Overview

TAMER is a machine learning approach that allows humans to train agents by providing real-time evaluative feedback (rewards and punishments). This web version ports the core TAMER algorithm from the original Java implementation.

## Quick Start

### Option 1: Using Python's built-in server

```bash
cd tamer-web
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Using Node.js

```bash
npx serve tamer-web
```

### Option 3: Any static file server

The application is purely client-side JavaScript and works with any static file server.

## How to Use

1. **Select an environment** from the dropdown (Loop Maze, Mountain Car, or Cart Pole)
2. **Press "Start"** or the **2** key to begin the simulation
3. **Give feedback** to the agent:
   - Press **/** (forward slash) for positive reward (+1) when the agent does something good
   - Press **Z** for negative reward (-1) when the agent does something bad
   - Press **?** for strong positive (+10)
   - Press **Shift+Z** for strong negative (-10)
4. **Watch the agent learn** from your feedback!

## Keyboard Controls

| Key | Action |
|-----|--------|
| `/` | Positive reward (+1) |
| `Z` | Negative reward (-1) |
| `?` | Strong positive (+10) |
| `Shift+Z` | Strong negative (-10) |
| `Space` | Toggle training mode |
| `2` | Start/Pause |
| `1` | Single step |
| `+/-` | Adjust speed |
| `R` | Reset agent |

## Environments

### Loop Maze
A simple gridworld navigation task. The agent must navigate from the start position to the goal. There's a short path and a long path around a loop - train the agent to take the short path!

### Mountain Car
A classic control task where an underpowered car must reach a goal on top of a hill. The car must build momentum by rocking back and forth. Train the agent to efficiently build momentum toward the goal.

### Cart Pole
Balance a pole on a moving cart. The episode ends if the pole falls too far or the cart hits the boundary. Train the agent to keep the pole balanced for as long as possible.

## Architecture

The codebase is organized as follows:

```
tamer-web/
├── index.html              # Main demo page with environment selector
├── css/
│   └── style.css           # Application styles
├── js/
│   ├── core/
│   │   ├── TamerAgent.js   # Main TAMER agent
│   │   ├── HLearner.js     # Human reward learner
│   │   └── CreditAssign.js # Temporal credit assignment
│   ├── models/
│   │   └── LinearModel.js  # Incremental gradient descent linear model
│   ├── features/
│   │   └── RBFFeatures.js  # Radial basis function features
│   ├── environments/
│   │   ├── Environment.js  # Base environment class
│   │   ├── LoopMaze.js     # Loop maze gridworld
│   │   ├── MountainCar.js  # Mountain car domain
│   │   └── CartPole.js     # Cart pole balancing
│   └── main.js             # Application entry point
└── demos/
    ├── loopmaze.html       # Dedicated Loop Maze demo
    ├── mountaincar.html    # Dedicated Mountain Car demo
    └── cartpole.html       # Dedicated Cart Pole demo
```

## Technical Details

### TAMER Algorithm

The TAMER algorithm learns a model of human reward preferences:

1. **Feature Generation**: Uses Radial Basis Functions (RBFs) to create features from state-action pairs
2. **Credit Assignment**: Distributes human feedback over recent state-action pairs using a temporal credit window
3. **Model Learning**: Updates a linear model via stochastic gradient descent
4. **Action Selection**: Chooses actions that maximize predicted human reward

### Credit Assignment Parameters

- **Credit Delay** (200ms): Time before the credit window starts (accounts for human reaction time)
- **Credit Window** (600ms): Duration during which past actions receive credit
- **Distribution Type**: How credit is distributed (uniform, previous-step weighted, or immediate)

## Original Project

This is a JavaScript port of the original Java TAMER project:
https://github.com/bradknox/tamerproject

## References

- Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *KCAP*.
- Knox, W. B., & Stone, P. (2010). Combining manual feedback with subsequent MDP reward signals for reinforcement learning. *AAMAS*.

## License

This project is released under the same license as the original TAMER project (Apache License 2.0).
