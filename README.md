# TAMER Web

TAMER Web is a JavaScript port of **TAMER**—*Training an Agent Manually via Evaluative Reinforcement*—meant for **interactive agent training directly in a web browser**. The goal is to keep the core ideas and mechanics faithful to the original Java TAMER codebase, while making it possible again to run demos and experiments online through people's browsers.

This port was created using Claude Code, with oversight and testing by TAMER's creator, Brad Knox.

If you are looking for the original Java implementation, see: [https://github.com/bradknox/tamerproject](https://github.com/bradknox/tamerproject)

## What TAMER is

TAMER is a framework for training an agent from **numeric feedback from a human**: a person watches the agent act and provides quick scalar feedback (approval/disapproval). Rather than relying on a hand-coded reward function from the environment, the agent learns a predictive model of what feedback it will receive for candidate actions, and then uses that model to decide what to do.

## Historic context

This approach is an early version of what is now commonly called **reinforcement learning from human feedback** (RLHF), reinforcement learning where the learning signal is provided by human judgments of an agent's behavior. RLHF as a problem class contrasts with methods such as traditional reinforcement learning (RL), using an environment’s built-in scoring rule for feedback, and imitation learning (IL), using demonstrations of desirable behavior. TAMER was the first *general-purpose* RLHF framework, applicable to the same tasks that RL and IL might be applied to. TAMER has been successfully applied to at least 10 different tasks by its original designers and others who have built upon it.

### Myopically pursuing near-term approval

A key design choice in TAMER is how it interprets the human’s feedback.

If an algorithm treats human feedback as if it were ordinary reinforcement-learning reward that be used for evaluation only in its accumulation (i.e., return)---as we did early on in initial development—--it generally does not work well. Instead, TAMER treats feedback as the human’s *full judgment of the long-term desirability of the agent’s recent behavior*. Under that assumption, the agent can be **myopic** in a principled way: it can simply choose actions that maximize predicted human approval, without explicitly planning far into the future.

TAMER's algorithm learns a function that predicts what feedback the human would give, and the agent greedily chooses the action with the highest predicted value.

In later work, we found that other early RLHF algorithms had been unintentionally tuned to be highly myopic through unusually agressive discounting of future reward (Knox et al., 2015). This characteristic had gone unnoticed even by their designers. Correspondingly, they often did not publish the discount factors in the corresponding research publications, though they graciously provided them by email to Knox. 

### Myopia in RLHF for LLMs

Contemporary RLHF for LLM fine-tuning is quite similar, trading the scalar feedback for preferences: humans compare or rate *whole* model responses, a reward model is trained to predict those judgments, and the policy is trained to maximize the reward model’s score.

Even though language models are trained for multi-turn interaction, a sequential decision-making problem, the learned reward model's output is optimized myopically, in a "bandit environment" as Ouyang et al. (2022) put it in their seminal paper on InstructGPT. In line with the TAMER framing, that scalar is not “environment reward” in the classical RL sense; it is a compact human judgment that already bakes in long-horizon considerations.

### TAMER+RL and “warm starts” from human feedback

In follow-up work on **TAMER+RL**, we explored how to combine:

* **human feedback**, which is information-rich but imperfect, and
* an environment’s **Markov decision process reward** (a task’s built-in scoring signal), which perfectly describes desirable behavior in the task, assuming return is the performance metric, but often is sample inefficient or even intractable to learn from.

One useful method that arose from the TAMER+RL work is a *warm start*: use human feedback to quickly get competent behavior, then let autonomous reinforcement learning take over (or blend the two signals) to refine performance. Building on the first TAMER+RL paper, Taylor et al. (2011) extended the concept of warm starts from RLHF to imitation learning, using human demonstrations to initialize a policy before learning from hard-coded environment reward.

This warm start method—bootstrap from human data, then improve with reinforcement learning—later became a common recipe in several high-profile algorithms such as AlphaGo (Silver et al. 2016), including pipelines that start from supervised learning on human data before switching to reinforcement learning.

## Relative strengths and weaknesses of TAMER

Strengths:
* Human trainers are giving their feedback live and immediately see the agent learn from it. This tightly interactive dynamic can make training a TAMER agent quite engaging, but it also allows the human trainer to learn to teach better by observing the effects of their feedback.
* Compared to standard behavior cloning (BC, the most common form of imitation learning), TAMER and other RLHF methods allow human labeling of states that the current policy visits, avoiding BC's problem of distribution shift that was made famous by the DAgger work by Ross et al.
* In comparison to traditional RL, learning from human input—whether preferences, scalar feedback, or demonstrations—leverages human knowledge about how to perform well at a sequential task. Contemporary LLM pipelines show this well, where demonstrations and preferences were the primary source of training data from ChatGPT until the introduction of thinking and reasoning LLMs, at which point RL from hard-coded reward was added to these forms of direct human input (and LLM-based simulations of that input).

Weaknesses:
* Giving feedback live introduces the need for TAMER's credit assignment, which cannot perfectly ascertain what actions the feedback targets. 
* When the action space is larger (i.e., the agent must choose from many actions), demonstrations allow singling out a single desired action. In contrast, TAMER and other RLHF methods involve feedback on whatever action(s) are taken (either live by the agent or in a dataset, as is the typical case for preference-based RLHF), resulting in a search over the large action space, a search that is sensitive to generalization.

## What this repository is

* a **browser-native** implementation of the TAMER loop (human feedback → learned predictor → action selection),
* suitable for **interactive demos, teaching, and online experimentation**, and
* a **faithful port** where it matters (including algorithmic minutia and the agent–human interaction loop).

This repo is not intended to be a production RLHF library for training large models.

## References

Referenced TAMER / TAMER+RL papers:

* Knox, W. B. & Stone, P. **“TAMER: Training an Agent Manually via Evaluative Reinforcement.”** (ICDL 2008). [https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf](https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf)
* Knox, W. B. & Stone, P. **“Combining Manual Feedback with Subsequent MDP Reward Signals for Reinforcement Learning.”** (AAMAS 2010). [https://www.cs.utexas.edu/~bradknox/papers/aamas10-knox.pdf](https://www.cs.utexas.edu/~bradknox/papers/aamas10-knox.pdf)
* Knox, W. B. & Stone, P. **“Reinforcement Learning from Simultaneous Human and MDP Reward.”** (AAMAS 2012). [https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS12-knox.pdf](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS12-knox.pdf)
* Knox, W. B. & Stone, P. **“Framing Reinforcement Learning from Human Reward: Reward Positivity, Temporal Discounting, Episodicity, and Performance.”** *Artificial Intelligence*, 225, 24–50. (August 2015). [https://doi.org/10.1016/j.artint.2015.03.002](https://doi.org/10.1016/j.artint.2015.03.002)


Other citations:

* Ouyang, L. et al. **“Training language models to follow instructions with human feedback.”** (InstructGPT, 2022). [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
* Silver, D. et al. **“Mastering the game of Go with deep neural networks and tree search.”** (AlphaGo, 2016). [https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
* Taylor, M. E., Suay, H. B. & Chernova, S. **“Integrating Reinforcement Learning with Human Demonstrations of Varying Ability.”** (AAMAS 2011). [https://www.ifaamas.org/Proceedings/aamas2011/papers/A5_R61.pdf](https://www.ifaamas.org/Proceedings/aamas2011/papers/A5_R61.pdf)


## Live Demo

Try it now: [Open the demo](index.html) (requires a local server - see Quick Start below)

## Quick Start

The application is purely client-side JavaScript and works with any static file server.

### Option 1: Python (recommended)

```bash
cd tamer-web
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Node.js

```bash
npx serve tamer-web
```

### Option 3: Any static file server

Use nginx, Apache, or any other static file server pointing to the `tamer-web` directory.

## How to Train an Agent

1. **Select an environment** from the dropdown menu
2. **Press Space** to enable training mode (indicator shows "Training: ON")
3. **Press "Start"** or the **2** key to begin the simulation
4. **Give feedback** to the agent as it acts:
   - Press **/** or **.** for positive reward (+1) when the agent does something good
   - Press **z** for negative reward (-1) when the agent does something bad
5. **Watch the agent learn** from your feedback in real-time!

The agent learns a model of your preferences and adjusts its behavior to maximize the feedback you provide.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `/` or `.` | Positive reward (+1) |
| `z` | Negative reward (-1) |
| `?` | Strong positive (+10) |
| `Shift+Z` | Strong negative (-10) |
| `Space` | Toggle training mode |
| `2` | Start/Pause simulation |
| `1` | Single step |
| `0` | Pause |
| `+`/`-` | Adjust simulation speed |
| `R` | Reset agent |

### Manual Control Mode

You can also control the agent directly to demonstrate desired behavior:

| Key | Action |
|-----|--------|
| `←` or `J` | Left / Push Left |
| `→` or `L` | Right / Push Right |
| `↑` or `I` | Up (4-action environments only) |
| `↓` or `K` | Down / Neutral |

Press `m` or click "Toggle Manual" to switch between TAMER mode (agent chooses actions) and Manual mode (you control the agent). Manual mode and training are mutually exclusive.

## Environments

### Tetris
The classic tile-matching game. The agent chooses piece placements (extended actions consisting of rotations and movements). Train the agent to clear lines efficiently!

**Actions:** Extended actions (piece placements)

**Note:** Manual control is not available for Tetris due to its use of extended actions.

### Loop Maze
A simple gridworld navigation task. The agent must navigate from the start position to the goal. There's a short path and a long path around a loop—train the agent to take the short path!

**Actions:** 4 directions (←↑→↓)

### Mountain Car
A classic control task where an underpowered car must reach a goal on top of a hill. The car cannot drive directly up the steep hill and must build momentum by rocking back and forth.

**Actions:** 3 (Left, Neutral, Right)

### Cart Pole
Balance a pole on a moving cart. The episode ends if the pole falls too far or the cart moves out of bounds. Train the agent to keep the pole balanced!

**Actions:** 2 (Push Left, Push Right)

### Acrobot
A two-link robot arm that must swing up to reach a target height. Only the joint between the two links is actuated.

**Actions:** 3 (Torque Left, None, Torque Right)

**Note:** Acrobot is a difficult task to teach by TAMER. Only expect to be able to effectively train a TAMER agent in Acrobot if you already know how to effectively control the acrobot.

### Robot Arm
A simple 2D robot arm that must reach target positions.

**Actions:** 4 directions

**Note:** Robot Arm is a difficult task to teach by TAMER. Only expect to be able to effectively train a TAMER agent in Robot Arm if you already know how to effectively control the robot arm.


## Architecture

```
tamer-web/
├── index.html                 # Intro/landing page with instructions
├── app.html                   # Main application with environment selector
├── css/
│   └── style.css              # Application styles
├── js/
│   ├── main.js                # Application entry point and TamerApp class
│   ├── core/
│   │   ├── TamerAgent.js      # Main TAMER agent implementation
│   │   ├── HLearner.js        # Human reward learner
│   │   ├── CreditAssign.js    # Temporal credit assignment
│   │   ├── HInfluence.js      # H-value influence on action selection
│   │   └── ActionSelect.js    # Action selection strategies
│   ├── agents/
│   │   ├── SarsaLambdaAgent.js    # SARSA(λ) RL agent
│   │   ├── TamerRLAgent.js        # TAMER+RL combined agent
│   │   ├── ImitationAgent.js      # Learning from demonstrations
│   │   └── ExtActionAgentWrap.js  # Wrapper for extended actions (Tetris)
│   ├── models/
│   │   └── LinearModel.js     # Incremental SGD linear model
│   ├── features/
│   │   ├── RBFFeatures.js     # Radial basis function features
│   │   └── TetrisFeatures.js  # Hand-crafted Tetris features
│   └── environments/
│       ├── Environment.js     # Base environment class
│       ├── LoopMaze.js        # Loop maze gridworld
│       ├── MountainCar.js     # Mountain car domain
│       ├── CartPole.js        # Cart pole balancing
│       ├── Acrobot.js         # Acrobot swing-up
│       ├── RobotArm.js        # 2D robot arm
│       └── tetris/            # Tetris implementation
│           ├── Tetris.js
│           ├── TetrisState.js
│           └── TetrisPiece.js
└── demos/                     # Standalone demo pages
    ├── loopmaze.html
    ├── mountaincar.html
    ├── cartpole.html
    ├── acrobot.html
    ├── robotarm.html
    └── tetris.html
```

## Technical Details

### TAMER Algorithm

The TAMER algorithm learns a model of human reward preferences Ĥ(s,a):

1. **Feature Generation**: State-action pairs are converted to feature vectors. This implementation uses Radial Basis Functions (RBFs) for continuous-state environments and hand-crafted features for Tetris. The Java codebase supports additional feature generators including tile coding and neural networks.

2. **Credit Assignment**: Human feedback is distributed over recent state-action pairs using a temporal credit window, accounting for human reaction time delay. Tetris uses a simpler "previous step" credit assignment since feedback applies to discrete piece placements.

3. **Model Learning**: A linear model is updated via stochastic gradient descent on the credited feedback.

4. **Action Selection**: The agent selects actions that maximize predicted human reward Ĥ(s,a).

### Credit Assignment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Credit Delay | 200ms | Time before credit window starts (human reaction time) |
| Credit Window | 600ms | Duration over which past actions receive credit |
| Distribution | Uniform | How credit is distributed within the window |

### Performance Optimizations

- **Fast Exp Approximation**: Uses Schraudolph (1999) algorithm for ~2.5x faster RBF computation
- **Adaptive Loop Strategy**: At normal speeds (≥16ms/step), renders every frame using requestAnimationFrame. At fast speeds (<16ms/step), runs multiple simulation steps per screen refresh and only renders the most recent state, allowing speeds up to thousands of steps per second.

## Differences from Java Version

This port aims to be functionally equivalent to the Java version. Key implementation notes:

- Uses JavaScript's native `Math` functions with Schraudolph fast exp approximation
- Rendering is handled via HTML5 Canvas instead of Java Swing
- All timing is based on `performance.now()` for high-resolution timestamps
- Environment parameters should match the Java defaults exactly

## Running Tests

```bash
node --experimental-vm-modules js/tests/test.js
```

## Original Project

This is a JavaScript port of the original Java TAMER project:
https://github.com/bradknox/tamerproject

The Java project includes additional features such as:
- RL-Glue compatibility for standardized agent-environment interaction
- Additional environments and experimental configurations
- TAMER+RL agents that combine human feedback with environmental rewards

## License

This project is released under the Apache License 2.0, the same license as the original TAMER project.

## Contributing

We are not anticipating accepting contributions, since this project is meant to exhibit past research implementations of the TAMER. That said, if you're the author of a peer-reviewed publication that implements a version of TAMER and want your work exhibited here as well, reach out.
