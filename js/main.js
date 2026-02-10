/**
 * TAMER Web - Main Application
 *
 * The main entry point that coordinates the environment, agent,
 * and user interface for interactive TAMER training.
 */
import { TamerAgent } from './core/TamerAgent.js?v=12';
import { LoopMaze } from './environments/LoopMaze.js';
import { MountainCar } from './environments/MountainCar.js';
import { CartPole } from './environments/CartPole.js';
import { Acrobot } from './environments/Acrobot.js';
import { Tetris, TetrisActions } from './environments/tetris/index.js?v=9';
import { RobotArm } from './environments/RobotArm.js';

// Additional agents
import { SarsaLambdaAgent } from './agents/SarsaLambdaAgent.js';
import { TamerRLAgent, CombinationMethods } from './agents/TamerRLAgent.js';
import { ImitationAgent } from './agents/ImitationAgent.js';
import { ExtActionAgentWrap } from './agents/ExtActionAgentWrap.js?v=9';

// Core components
import { ActionSelect } from './core/ActionSelect.js';
import { HInfluence } from './core/HInfluence.js';

// Feature generators
import { TetrisFeatures } from './features/TetrisFeatures.js?v=9';

// Available environments
const ENVIRONMENTS = {
    'loopmaze': LoopMaze,
    'mountaincar': MountainCar,
    'cartpole': CartPole,
    'acrobot': Acrobot,
    'tetris': Tetris,
    'robotarm': RobotArm
};

// Default timestep durations in ms - from Java TAMER project
// Continuous control environments use 200ms (RunLocalExperiment.java default)
// LoopMaze uses 800ms (D05LoopMazeTamer.java)
// Tetris uses 150ms (TetrisTamerExp.java)
const DEFAULT_STEP_DURATIONS = {
    'loopmaze': 800,
    'mountaincar': 200,
    'cartpole': 200,
    'acrobot': 200,
    'tetris': 150,
    'robotarm': 200
};

export class TamerApp {
    constructor(canvasId, options = {}) {
        // Get canvas and context
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element '${canvasId}' not found`);
        }
        this.ctx = this.canvas.getContext('2d');

        // Configuration
        this.envName = options.environment ?? 'loopmaze';
        // Use environment-specific default step duration if not explicitly provided
        this.stepDurationMs = options.stepDurationMs ?? DEFAULT_STEP_DURATIONS[this.envName] ?? 500;
        this.autoStart = options.autoStart ?? false;

        // Store whether user explicitly set various options
        this._userSetBasisFcns = options.basisFcnsPerDim !== undefined;
        this._userSetStepSize = options.stepSize !== undefined;
        this._userSetCreditDelayMs = options.creditDelayMs !== undefined;
        this._userSetCreditWindowMs = options.creditWindowMs !== undefined;
        this._userSetCreditDistType = options.creditDistType !== undefined;

        // Agent options - default basisFcnsPerDim=40 matches Java
        this.agentOptions = {
            epsilon: options.epsilon ?? 0.0,
            stepSize: options.stepSize ?? 0.05,  // EXACT from Java Params.java default
            basisFcnsPerDim: options.basisFcnsPerDim ?? 40,
            creditDelayMs: options.creditDelayMs ?? 200,
            creditWindowMs: options.creditWindowMs ?? 600,
            creditDistType: options.creditDistType ?? 'uniform'
        };

        // State
        this.running = false;
        this.paused = false;
        this.lastStepTime = 0;
        this.animationFrameId = null;
        this.timeoutId = null;  // For fast simulation mode

        // Manual control mode
        this.manualControlMode = false;
        this.lastUserActI = -1;  // Last user action input (-1 = none)

        // Episode end pause - EXACT from Java RunLocalExperiment.PAUSE_DUR_AFTER_EP
        // Allows user to give feedback for final actions of the episode
        this.pauseDurAfterEp = options.pauseDurAfterEp ?? 1000;  // 1 second default
        this.inEpisodeEndPause = false;
        this.episodeEndPauseStart = 0;

        // Statistics
        this.totalEpisodes = 0;
        this.stepsThisEp = 0;  // Steps in current episode (reset each episode)
        this.rewardsGiven = { positive: 0, negative: 0 };

        // Tetris H-value tracking for bar visualization
        // Initialize with tiny values so 0 starts centered
        this.allTimeMinH = -0.000001;
        this.allTimeMaxH = 0.000001;

        // Track current Tetris placement being executed
        this._currentTetrisPlacement = null;
        // Cache placements for the current piece to avoid recalculating during execution
        this._cachedTetrisPlacements = null;
        this._cachedTetrisBlockId = null;
        this._cachedTetrisObs = null;

        // UI callbacks
        this.onStepCallback = null;
        this.onEpisodeEndCallback = null;
        this.onRewardCallback = null;

        // Initialize
        this._init();
    }

    /**
     * Initialize the application
     */
    _init() {
        // Create environment
        const EnvClass = ENVIRONMENTS[this.envName];
        if (!EnvClass) {
            throw new Error(`Unknown environment: ${this.envName}`);
        }
        this.env = new EnvClass();

        // Environment-specific configurations - EXACT from Java Params.java getParams()
        // Each environment has specific settings that override the defaults
        const ENV_SPECIFIC_CONFIG = {
            'loopmaze': {
                // EXACT from Java Params.java lines 146-157 for "Loop-Maze"
                basisFcnsPerDim: 6,
                relWidth: 0.05,
                stepSize: 0.2,
                creditDelayMs: 150,   // Java: creditDelay = 0.15
                creditWindowMs: 250,  // Java: windowSize = 0.25
                creditDistType: 'uniform'
            },
            'mountaincar': {
                // EXACT from Java Params.java setPyMCParams() lines 332-347
                basisFcnsPerDim: 40,
                relWidth: 0.08,
                stepSize: 0.05  // Default from Params.java
            },
            'cartpole': {
                // EXACT from Java Params.java line 307-308 for HInfluence/CartPole
                basisFcnsPerDim: 8,
                relWidth: 0.08
            },
            'acrobot': {
                // Similar to CartPole for high-dimensional continuous space
                basisFcnsPerDim: 8,
                relWidth: 0.08
            },
            'robotarm': {
                // Similar settings for continuous control
                basisFcnsPerDim: 8,
                relWidth: 0.08
            }
        };

        // Apply environment-specific overrides
        const envAgentOptions = { ...this.agentOptions };
        const envConfig = ENV_SPECIFIC_CONFIG[this.envName];
        if (envConfig) {
            // Only apply if user didn't explicitly set these values
            if (!this._userSetBasisFcns && envConfig.basisFcnsPerDim !== undefined) {
                envAgentOptions.basisFcnsPerDim = envConfig.basisFcnsPerDim;
            }
            if (envConfig.relWidth !== undefined) {
                envAgentOptions.relWidth = envConfig.relWidth;
            }
            if (envConfig.stepSize !== undefined && !this._userSetStepSize) {
                envAgentOptions.stepSize = envConfig.stepSize;
            }
            if (envConfig.creditDelayMs !== undefined && !this._userSetCreditDelayMs) {
                envAgentOptions.creditDelayMs = envConfig.creditDelayMs;
            }
            if (envConfig.creditWindowMs !== undefined && !this._userSetCreditWindowMs) {
                envAgentOptions.creditWindowMs = envConfig.creditWindowMs;
            }
            if (envConfig.creditDistType !== undefined && !this._userSetCreditDistType) {
                envAgentOptions.creditDistType = envConfig.creditDistType;
            }
        }

        // Use ExtActionAgentWrap with specialized settings for Tetris
        // EXACT from Java TetrisTamerExpHelper.setTamerAgentParams()
        if (this.envName === 'tetris') {
            const tetrisAgentOptions = {
                FeatGenClass: TetrisFeatures,
                epsilon: 0.0,
                // EXACT from Java: stepSize = 0.000005 / 47
                stepSize: 0.000005 / 47,
                // EXACT from Java: distClass = "previousStep"
                creditDistType: 'previousStep',
                // Credit assignment parameters from Java
                creditDelayMs: 200,
                creditWindowMs: 600,
                // EXACT from Java Params.java line 166: extrapolateFutureRew = false for Tetris
                extrapolateFutureRew: false
            };

            // Create ExtActionAgentWrap for Tetris (handles piece placements as extended actions)
            console.log('Creating ExtActionAgentWrap for Tetris...');
            this.agent = new ExtActionAgentWrap(this.env, {
                coreAgentClass: TamerAgent,
                coreAgentOptions: tetrisAgentOptions
            });
            console.log('Calling agent.init()...');
            this.agent.init();
            console.log('Tetris agent initialized');
        } else {
            // Use specialized feature generator for other environments
            // Create agent
            this.agent = new TamerAgent(this.env, envAgentOptions);
        }

        // Set up keyboard input
        this._boundHandleKeyDown = (e) => this._handleKeyDown(e);
        document.addEventListener('keydown', this._boundHandleKeyDown);

        // Get action values canvas if available
        this.actionValuesCanvas = document.getElementById('action-values-canvas');
        this.actionValuesCtx = this.actionValuesCanvas ? this.actionValuesCanvas.getContext('2d') : null;

        // Get Tetris thumbnails container
        this.tetrisThumbnailsContainer = document.getElementById('tetris-thumbnails-container');

        // Set up canvas resize handling - use requestAnimationFrame to ensure layout is complete
        this._boundResizeCanvas = () => this._resizeCanvas();
        window.addEventListener('resize', this._boundResizeCanvas);
        // Initial resize after layout settles
        requestAnimationFrame(() => {
            this._resizeCanvas();
            this._render();
        });

        // Initial render
        this._render();
    }

    /**
     * Handle keyboard input
     */
    _handleKeyDown(event) {
        // Allow all browser shortcuts (CMD/Ctrl + key) to pass through completely
        if (event.metaKey || event.ctrlKey) {
            return; // Don't handle any shortcuts with CMD/Ctrl
        }

        const key = event.key.toLowerCase();
        const time = performance.now();

        switch (key) {
            case '/':
            case '.':
                // Positive reward (+1)
                event.preventDefault();
                this._giveReward(1.0, time);
                break;

            case 'z':
                // Negative reward (-1)
                event.preventDefault();
                this._giveReward(-1.0, time);
                break;

            case '?':
                // Strong positive reward (+10)
                event.preventDefault();
                this._giveReward(10.0, time);
                break;

            case 'shift':
                if (event.key === 'Z') {
                    // Strong negative reward (-10)
                    event.preventDefault();
                    this._giveReward(-10.0, time);
                }
                break;

            case ' ':
                // Toggle training mode
                event.preventDefault();
                const isTraining = this.agent.toggleTraining();
                this._updateTrainingIndicator(isTraining);

                // Manual mode and training are mutually exclusive
                if (isTraining && this.manualControlMode) {
                    this.manualControlMode = false;
                    this.lastUserActI = -1;
                    this._updateModeIndicator(false);
                }
                break;

            case '1':
                // Single step
                event.preventDefault();
                if (!this.running) {
                    this._singleStep();
                }
                break;

            case '2':
                // Start/stop
                event.preventDefault();
                this.toggle();
                break;

            case '+':
            case '=':
                // Speed up (logarithmic steps for fine control at low values)
                event.preventDefault();
                if (this.stepDurationMs > 100) {
                    this.stepDurationMs = Math.max(100, this.stepDurationMs - 50);
                } else if (this.stepDurationMs > 10) {
                    this.stepDurationMs = Math.max(10, this.stepDurationMs - 10);
                } else if (this.stepDurationMs > 1) {
                    this.stepDurationMs = Math.max(1, this.stepDurationMs - 1);
                } else if (this.stepDurationMs > 0.1) {
                    this.stepDurationMs = Math.max(0.1, this.stepDurationMs - 0.1);
                } else if (this.stepDurationMs > 0.01) {
                    this.stepDurationMs = Math.max(0.01, this.stepDurationMs - 0.01);
                } else {
                    this.stepDurationMs = Math.max(0.001, this.stepDurationMs - 0.001);
                }
                this._updateSpeedIndicator();
                break;

            case '-':
            case '_':
                // Slow down (logarithmic steps)
                event.preventDefault();
                if (this.stepDurationMs < 0.01) {
                    this.stepDurationMs = Math.min(0.01, this.stepDurationMs + 0.001);
                } else if (this.stepDurationMs < 0.1) {
                    this.stepDurationMs = Math.min(0.1, this.stepDurationMs + 0.01);
                } else if (this.stepDurationMs < 1) {
                    this.stepDurationMs = Math.min(1, this.stepDurationMs + 0.1);
                } else if (this.stepDurationMs < 10) {
                    this.stepDurationMs = Math.min(10, this.stepDurationMs + 1);
                } else if (this.stepDurationMs < 100) {
                    this.stepDurationMs = Math.min(100, this.stepDurationMs + 10);
                } else {
                    this.stepDurationMs = Math.min(2000, this.stepDurationMs + 50);
                }
                this._updateSpeedIndicator();
                break;

            case '0':
                // Pause
                event.preventDefault();
                this.pause();
                break;

            case 'r':
                // Reset
                event.preventDefault();
                this.reset();
                break;

            case 'm':
                // Toggle manual control mode
                event.preventDefault();
                this.toggleManualControlMode();
                break;

            // Manual control keys - EXACT from Java ImitationAgent
            case 'j':
            case 'arrowleft':
                // Left/accelerate left
                event.preventDefault();
                this._handleManualAction('left');
                break;
            case 'k':
            case 'arrowdown':
                // Down/neutral/no action
                event.preventDefault();
                this._handleManualAction('down');
                break;
            case 'l':
            case 'arrowright':
                // Right/accelerate right
                event.preventDefault();
                this._handleManualAction('right');
                break;
            case 'i':
            case 'arrowup':
                // Up
                event.preventDefault();
                this._handleManualAction('up');
                break;
        }
    }

    /**
     * Handle manual control action key press - EXACT from Java ImitationAgent
     * Maps keyboard keys to actions based on environment
     * Accepts: 'left', 'right', 'up', 'down' (from arrow keys or j/k/l/i)
     */
    _handleManualAction(direction) {
        if (!this.manualControlMode) return;

        // Environment-specific key mappings - EXACT from Java ImitationAgent
        if (this.envName === 'loopmaze') {
            // 0=right, 1=left, 2=down, 3=up (from LoopMaze action labels)
            if (direction === 'left') this.lastUserActI = 1;
            else if (direction === 'down') this.lastUserActI = 2;
            else if (direction === 'right') this.lastUserActI = 0;
            else if (direction === 'up') this.lastUserActI = 3;
        } else if (this.envName === 'cartpole') {
            // 0=left, 1=right
            if (direction === 'left') this.lastUserActI = 0;
            else if (direction === 'right') this.lastUserActI = 1;
        } else if (this.envName === 'mountaincar') {
            // 0=left, 1=neutral, 2=right
            if (direction === 'left') this.lastUserActI = 0;
            else if (direction === 'down') this.lastUserActI = 1;  // neutral
            else if (direction === 'right') this.lastUserActI = 2;
        } else if (this.envName === 'acrobot') {
            // 0=left, 1=none, 2=right
            if (direction === 'left') this.lastUserActI = 0;
            else if (direction === 'down') this.lastUserActI = 1;  // no torque
            else if (direction === 'right') this.lastUserActI = 2;
        } else if (this.envName === 'robotarm') {
            // Default 4-action mapping for robot arm
            if (direction === 'left') this.lastUserActI = 1;
            else if (direction === 'down') this.lastUserActI = 2;
            else if (direction === 'right') this.lastUserActI = 0;
            else if (direction === 'up') this.lastUserActI = 3;
        }
        // Note: Tetris uses extended actions, manual control not supported
    }

    /**
     * Give a reward signal
     */
    _giveReward(reward, time) {
        if (!this.running && !this.paused) return;

        // Only process rewards when training is enabled
        if (!this.agent.isTraining()) {
            this._flashTrainingIndicator();
            return;
        }

        this.agent.processHumanReward(reward, time);

        if (reward > 0) {
            this.rewardsGiven.positive++;
        } else {
            this.rewardsGiven.negative++;
        }

        // Visual feedback
        this._flashRewardIndicator(reward > 0 ? 'positive' : 'negative');

        if (this.onRewardCallback) {
            this.onRewardCallback(reward, time);
        }
    }

    /**
     * Start or resume running
     */
    start() {
        console.log('TamerApp.start() called, running:', this.running);
        if (this.running) return;

        this.running = true;
        this.paused = false;

        // Start first episode if needed
        console.log('Checking agent.currentObs:', this.agent.currentObs);
        if (this.agent.currentObs === null) {
            console.log('Starting first episode...');
            try {
                this.agent.startEpisode();
                this.totalEpisodes++;
                this.stepsThisEp = 0;
                console.log('Episode started, totalEpisodes:', this.totalEpisodes);
            } catch (e) {
                console.error('Error starting episode:', e);
            }
        }

        this.lastStepTime = performance.now();
        console.log('Starting run loop...');
        this._runLoop();
    }

    /**
     * Stop running
     */
    stop() {
        this.running = false;
        this.paused = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        // Clear canvas to prevent flashing of previous environment state
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Fully clean up this instance (stop + remove event listeners)
     */
    destroy() {
        this.stop();
        document.removeEventListener('keydown', this._boundHandleKeyDown);
        window.removeEventListener('resize', this._boundResizeCanvas);
    }

    /**
     * Pause running
     */
    pause() {
        this.paused = true;
        this.running = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        // Render current state to ensure display is correct while paused
        this._render();
    }

    /**
     * Toggle running state
     */
    toggle() {
        if (this.running) {
            this.pause();
        } else {
            this.start();
        }
    }

    /**
     * Reset the application
     */
    reset() {
        this.stop();
        this.agent.reset();
        this.env.start();
        this.totalEpisodes = 0;
        this.stepsThisEp = 0;
        this.rewardsGiven = { positive: 0, negative: 0 };
        // Reset manual control state
        this.lastUserActI = -1;
        // Reset H-value bounds to tiny values so 0 starts centered
        this.allTimeMinH = -0.000001;
        this.allTimeMaxH = 0.000001;
        this._lastTetrisObs = null;
        this._currentTetrisPlacement = null;
        this._cachedTetrisPlacements = null;
        this._cachedTetrisBlockId = null;
        this._cachedTetrisObs = null;
        // Clear canvas before rendering to prevent flashing
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this._render();
    }

    /**
     * Take a single step
     */
    _singleStep() {
        if (this.agent.currentObs === null) {
            this.agent.startEpisode();
            this.totalEpisodes++;
            this.stepsThisEp = 0;
        }

        const time = performance.now();

        // In manual mode, use user action or default to 0
        let result;
        if (this.manualControlMode && this.envName !== 'tetris') {
            // Note: lastUserActI persists across steps (matches Java behavior for held keys)
            const action = this.lastUserActI !== -1 ? this.lastUserActI : 0;

            const envResult = this.env.step(action);
            this.stepsThisEp++;

            if (this.agent.isTraining()) {
                this.agent.processManualStep(action, envResult.obs, time);
            }

            result = {
                action: action,
                obs: envResult.obs,
                reward: envResult.reward,
                terminal: envResult.terminal
            };
        } else {
            result = this.agent.step(time);
            this.stepsThisEp++;
        }

        if (result.terminal) {
            this.agent.endEpisode();
            // Reset manual control action at episode end - EXACT from Java ImitationAgent behavior
            this.lastUserActI = -1;

            if (this.onEpisodeEndCallback) {
                this.onEpisodeEndCallback(this.env.getEpisodeStats());
            }

            // Start new episode
            this.agent.startEpisode();
            this.totalEpisodes++;
            this.stepsThisEp = 0;
        }

        if (this.onStepCallback) {
            this.onStepCallback(result);
        }

        this._render();
    }

    /**
     * Main run loop - chooses appropriate timing strategy
     */
    _runLoop() {
        if (!this.running) {
            console.log('_runLoop: not running, returning');
            return;
        }

        // Use different strategies based on step duration
        // requestAnimationFrame is limited to ~60Hz (16ms), so use setTimeout for faster speeds
        if (this.stepDurationMs >= 16) {
            this._runLoopRAF();
        } else {
            this._runLoopFast();
        }
    }

    /**
     * Run loop using requestAnimationFrame - for normal speeds (>= 16ms/step)
     */
    _runLoopRAF() {
        if (!this.running) return;

        // Check if we should switch to fast loop (user changed speed)
        if (this.stepDurationMs < 16) {
            this._runLoopFast();
            return;
        }

        const currentTime = performance.now();

        // Handle episode end pause - EXACT from Java RunLocalExperiment
        if (this.inEpisodeEndPause) {
            const pauseElapsed = currentTime - this.episodeEndPauseStart;
            if (pauseElapsed >= this.pauseDurAfterEp) {
                this.inEpisodeEndPause = false;
                this.agent.startEpisode();
                this.totalEpisodes++;
                this.stepsThisEp = 0;
                this.lastStepTime = currentTime;
            }
            this._render();
            this.animationFrameId = requestAnimationFrame(() => this._runLoopRAF());
            return;
        }

        const elapsed = currentTime - this.lastStepTime;

        if (elapsed >= this.stepDurationMs) {
            this._doStep(currentTime);
            this.lastStepTime = currentTime;
        }

        this._render();
        this.animationFrameId = requestAnimationFrame(() => this._runLoopRAF());
    }

    /**
     * Run loop using setTimeout - for fast simulation (< 16ms/step)
     * Runs multiple steps per frame and renders periodically
     */
    _runLoopFast() {
        if (!this.running) return;

        // Check if we should switch to RAF loop (user slowed down)
        if (this.stepDurationMs >= 16) {
            this._runLoopRAF();
            return;
        }

        const currentTime = performance.now();

        // Handle episode end pause
        if (this.inEpisodeEndPause) {
            const pauseElapsed = currentTime - this.episodeEndPauseStart;
            if (pauseElapsed >= this.pauseDurAfterEp) {
                this.inEpisodeEndPause = false;
                this.agent.startEpisode();
                this.totalEpisodes++;
                this.stepsThisEp = 0;
                this.lastStepTime = currentTime;
            } else {
                this._render();
                this.timeoutId = setTimeout(() => this._runLoopFast(), 0);
                return;
            }
        }

        // Calculate how many steps to run per render (~60fps = 16ms)
        // Run more steps for faster simulation
        const stepsPerFrame = Math.max(1, Math.floor(16 / Math.max(0.001, this.stepDurationMs)));

        for (let i = 0; i < stepsPerFrame && this.running && !this.inEpisodeEndPause; i++) {
            this._doStep(performance.now());
        }

        this._render();

        // Schedule next batch using setTimeout(0) for maximum speed
        this.timeoutId = setTimeout(() => this._runLoopFast(), 0);
    }

    /**
     * Execute a single step
     */
    _doStep(currentTime) {
        try {
            // In manual mode, get action from user input or use default (0)
            let result;
            if (this.manualControlMode && this.envName !== 'tetris') {
                // Use user action if provided, otherwise default to action 0 - EXACT from Java ImitationAgent line 234
                // Note: lastUserActI persists across steps (NOT reset here) - this matches Java behavior
                // where held keys work because the action persists until a new key is pressed or episode ends
                const action = this.lastUserActI !== -1 ? this.lastUserActI : 0;

                // Manually step with the chosen action
                const envResult = this.env.step(action);
                this.stepsThisEp++;

                // If training is on in manual mode, still let the agent learn
                if (this.agent.isTraining()) {
                    // Process the step through the agent for learning
                    this.agent.processManualStep(action, envResult.obs, currentTime);
                }

                result = {
                    action: action,
                    obs: envResult.obs,
                    reward: envResult.reward,
                    terminal: envResult.terminal
                };
            } else {
                result = this.agent.step(currentTime);
                this.stepsThisEp++;
            }

            if (result.terminal) {
                this.agent.endEpisode();
                // Reset manual control action at episode end - EXACT from Java ImitationAgent behavior
                this.lastUserActI = -1;

                if (this.onEpisodeEndCallback) {
                    this.onEpisodeEndCallback(this.env.getEpisodeStats());
                }

                if (this.pauseDurAfterEp > 0) {
                    this.inEpisodeEndPause = true;
                    this.episodeEndPauseStart = currentTime;
                } else {
                    this.agent.startEpisode();
                    this.totalEpisodes++;
                    this.stepsThisEp = 0;
                }
            }

            if (this.onStepCallback) {
                this.onStepCallback(result);
            }
        } catch (e) {
            console.error('Error in step:', e);
        }
    }

    /**
     * Resize canvas to fit within container while maintaining square aspect ratio
     */
    _resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        // Get container's content box dimensions
        const containerRect = container.getBoundingClientRect();
        const containerStyle = getComputedStyle(container);
        const paddingX = parseFloat(containerStyle.paddingLeft) + parseFloat(containerStyle.paddingRight);
        const paddingY = parseFloat(containerStyle.paddingTop) + parseFloat(containerStyle.paddingBottom);
        const borderX = parseFloat(containerStyle.borderLeftWidth) + parseFloat(containerStyle.borderRightWidth);
        const borderY = parseFloat(containerStyle.borderTopWidth) + parseFloat(containerStyle.borderBottomWidth);

        // Get the env-selector height if it exists (includes its margin)
        const envSelector = container.querySelector('.env-selector');
        let envSelectorHeight = 0;
        if (envSelector) {
            const envStyle = getComputedStyle(envSelector);
            envSelectorHeight = envSelector.offsetHeight +
                parseFloat(envStyle.marginTop) + parseFloat(envStyle.marginBottom);
        }

        // Calculate available space for the canvas
        // Account for canvas border (2px total from 1px each side)
        const canvasBorder = 2;
        const availableWidth = containerRect.width - paddingX - borderX - canvasBorder;
        const availableHeight = containerRect.height - paddingY - borderY - envSelectorHeight - canvasBorder;

        // Use the smaller dimension to maintain square aspect ratio
        // Subtract safety margin to account for any browser rendering differences
        const size = Math.floor(Math.min(availableWidth, availableHeight)) - 4;

        if (size > 0 && (this.canvas.width !== size || this.canvas.height !== size)) {
            // Set both the drawing buffer size and CSS display size
            this.canvas.width = size;
            this.canvas.height = size;
            this.canvas.style.width = size + 'px';
            this.canvas.style.height = size + 'px';
            // Re-render after resize
            if (this.env) {
                this._render();
            }
        }
    }

    /**
     * Render the current state
     */
    _render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Render environment
        this.env.render(this.ctx, this.canvas.width, this.canvas.height);

        // Render action values overlay if agent has learned something
        this._renderActionValues();
    }

    /**
     * Render action value predictions to separate canvas
     */
    _renderActionValues() {
        // Use separate canvas if available
        if (!this.actionValuesCtx) return;

        // For Tetris, use specialized thumbnail renderer
        if (this.envName === 'tetris') {
            this._renderTetrisActionValues();
            return;
        }

        // For other environments, hide Tetris container and show bar chart
        if (this.tetrisThumbnailsContainer) {
            this.tetrisThumbnailsContainer.style.display = 'none';
        }
        this.actionValuesCanvas.style.display = 'block';

        const values = this.agent.getActionValues();
        const ctx = this.actionValuesCtx;
        const canvas = this.actionValuesCanvas;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (values.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('No predictions yet', canvas.width / 2, canvas.height / 2);
            return;
        }

        const labels = this.env.getActionLabels();

        // Find min and max values across all actions for uniform scaling
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal;

        // Calculate dimensions
        const barWidth = Math.floor(canvas.width / values.length);
        const topMargin = 15;     // Space for value text above bars
        const bottomMargin = 18;  // Space for labels
        const barAreaHeight = canvas.height - topMargin - bottomMargin;

        // Calculate where the zero line should be within the bar area
        // If range is 0, center the baseline
        let baseY;
        if (range < 0.0001) {
            baseY = topMargin + barAreaHeight / 2;
        } else {
            // baseY is positioned proportionally: how far maxVal is from the top
            // If all positive, baseY near bottom; if all negative, baseY near top
            baseY = topMargin + (maxVal / range) * barAreaHeight;
        }

        // Clamp baseY to valid range
        baseY = Math.max(topMargin + 5, Math.min(canvas.height - bottomMargin - 5, baseY));

        ctx.font = '11px Arial';
        ctx.textAlign = 'center';

        for (let i = 0; i < values.length; i++) {
            const x = i * barWidth + barWidth / 2;

            // Calculate bar height using uniform scale
            let barHeight = 0;
            if (range >= 0.0001) {
                barHeight = (values[i] / range) * barAreaHeight;
            }

            // Draw bar
            if (values[i] >= 0) {
                ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
                ctx.fillRect(x - barWidth/3, baseY - barHeight, barWidth * 2/3, Math.max(1, barHeight));
            } else {
                ctx.fillStyle = 'rgba(244, 67, 54, 0.8)';
                ctx.fillRect(x - barWidth/3, baseY, barWidth * 2/3, Math.max(1, -barHeight));
            }

            // Draw value above/below bar
            ctx.fillStyle = '#000';
            ctx.font = '10px Arial';
            const valueY = values[i] >= 0 ? baseY - barHeight - 3 : baseY - barHeight + 10;
            ctx.fillText(values[i].toFixed(3), x, Math.max(12, Math.min(canvas.height - bottomMargin - 2, valueY)));
            ctx.font = '11px Arial';

            // Draw label at bottom
            ctx.fillStyle = '#333';
            ctx.fillText(labels[i], x, canvas.height - 5);
        }

        // Draw baseline (zero line)
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, baseY);
        ctx.lineTo(canvas.width, baseY);
        ctx.stroke();
    }

    /**
     * Render Tetris action values as mini-board thumbnails
     * Each thumbnail shows the board with ghost piece at placement position
     * with an H value bar below
     */
    _renderTetrisActionValues() {
        if (!this.tetrisThumbnailsContainer) return;

        // Always hide bar chart and show thumbnails container when in Tetris mode
        this.actionValuesCanvas.style.display = 'none';
        this.tetrisThumbnailsContainer.style.display = 'flex';

        // Get current observation - use cached obs during pause if available
        const obs = this.agent.currentObs || this._lastTetrisObs;
        if (!obs) {
            this.tetrisThumbnailsContainer.innerHTML = '<span style="color: #666; font-size: 11px;">No state yet</span>';
            return;
        }

        // Cache observation for use during pause
        this._lastTetrisObs = obs;

        // Get feature generator and current block ID
        const featGen = this.agent.coreAgent.featGen;
        featGen.setStateFromObs(obs);
        const currentBlockId = featGen.gameState.currentBlockId;

        // Check if we need to regenerate placements (new piece or first time)
        const needNewPlacements = !this._cachedTetrisPlacements ||
            this._cachedTetrisBlockId !== currentBlockId ||
            this.agent.currExtActI === 0;

        let placements;
        if (needNewPlacements) {
            // Get extended actions (placements) from feature generator
            const extActions = featGen.getPossActions(obs);

            if (!extActions || extActions.length === 0) {
                this.tetrisThumbnailsContainer.innerHTML = '<span style="color: #666; font-size: 11px;">No placements available</span>';
                return;
            }

            // Get H values for all actions
            const hValues = this.agent.getActionValues();

            // Create placement data with average column position for sorting
            placements = extActions.map((extAction, idx) => {
                // Get the piece shape at the final position
                const state = featGen.gameState;
                const piece = state.possibleBlocks[state.currentBlockId].getShape(extAction.currentRotation);

                // Calculate average column position of the 4 cells
                let colSum = 0;
                let cellCount = 0;
                for (let x = 0; x < piece.length; x++) {
                    for (let y = 0; y < piece[0].length; y++) {
                        if (piece[x][y] !== 0) {
                            colSum += (extAction.currentX + x);
                            cellCount++;
                        }
                    }
                }
                const avgCol = cellCount > 0 ? colSum / cellCount : 0;

                return {
                    extAction,
                    hValue: hValues[idx] || 0,
                    avgCol,
                    blockId: state.currentBlockId,
                    rotation: extAction.currentRotation
                };
            });

            // Sort by average column position (left to right)
            placements.sort((a, b) => a.avgCol - b.avgCol);

            // Cache the placements
            this._cachedTetrisPlacements = placements;
            this._cachedTetrisBlockId = currentBlockId;
            this._cachedTetrisObs = obs.slice(0, 200);  // Cache just the board state
        } else {
            // Use cached placements
            placements = this._cachedTetrisPlacements;
        }

        // Update all-time min/max H values
        const hValuesOnly = placements.map(p => p.hValue);
        const currentMinH = Math.min(...hValuesOnly);
        const currentMaxH = Math.max(...hValuesOnly);
        this.allTimeMinH = Math.min(this.allTimeMinH, currentMinH);
        this.allTimeMaxH = Math.max(this.allTimeMaxH, currentMaxH);

        // Track the currently executing placement
        // Update cached placement when a new extended action starts (currExtActI === 0)
        if (this.agent.currExtActI === 0 && this.agent.currExtendedAction) {
            // Find the placement that matches the current extended action
            for (const p of placements) {
                if (this._extActionsMatch(p.extAction.actList, this.agent.currExtendedAction)) {
                    this._currentTetrisPlacement = {
                        x: p.extAction.currentX,
                        y: p.extAction.currentY,
                        rotation: p.extAction.currentRotation,
                        blockId: p.blockId
                    };
                    break;
                }
            }
        }

        // Clear container
        this.tetrisThumbnailsContainer.innerHTML = '';

        // Thumbnail dimensions
        const cellSize = 2;  // pixels per cell
        const boardWidth = 10;
        const boardHeight = 20;
        const thumbWidth = boardWidth * cellSize;
        const thumbHeight = boardHeight * cellSize;
        const barHeight = 4;
        const textHeight = 10;  // Space for H value text

        // Calculate where 0 falls within the all-time range
        const hRange = this.allTimeMaxH - this.allTimeMinH;
        let zeroX;
        if (hRange < 0.0000001) {
            zeroX = thumbWidth / 2;  // Center if no range
        } else {
            // Position of 0 within [allTimeMinH, allTimeMaxH]
            zeroX = ((0 - this.allTimeMinH) / hRange) * thumbWidth;
            // Clamp to valid range
            zeroX = Math.max(1, Math.min(thumbWidth - 1, zeroX));
        }

        // Find the argmax placement (highest H value)
        let maxHValue = -Infinity;
        let argmaxPlacement = null;
        for (const p of placements) {
            if (p.hValue > maxHValue) {
                maxHValue = p.hValue;
                argmaxPlacement = p;
            }
        }

        // Render each placement as a thumbnail
        for (const placement of placements) {
            // Check if this matches the cached current placement
            const isCurrentPlacement = this._currentTetrisPlacement &&
                placement.extAction.currentX === this._currentTetrisPlacement.x &&
                placement.extAction.currentY === this._currentTetrisPlacement.y &&
                placement.extAction.currentRotation === this._currentTetrisPlacement.rotation &&
                placement.blockId === this._currentTetrisPlacement.blockId;

            // Check if this is the argmax placement
            const isArgmax = placement === argmaxPlacement;

            const div = document.createElement('div');
            div.className = 'tetris-thumbnail';
            if (isCurrentPlacement && isArgmax) {
                // Both current and argmax - show both colors
                div.style.outline = '2px solid #2196F3';
                div.style.boxShadow = '0 0 0 4px #f44336';
            } else if (isCurrentPlacement) {
                div.style.outline = '2px solid #2196F3';
                div.style.outlineOffset = '1px';
            } else if (isArgmax) {
                div.style.outline = '2px solid #f44336';
                div.style.outlineOffset = '1px';
            }

            // Create canvas for thumbnail
            const canvas = document.createElement('canvas');
            canvas.width = thumbWidth;
            canvas.height = thumbHeight;
            const ctx = canvas.getContext('2d');

            // Render the thumbnail using cached board state for consistency
            const renderObs = this._cachedTetrisObs || obs;
            this._renderTetrisThumbnail(ctx, renderObs, placement, cellSize);

            div.appendChild(canvas);

            // Create H value bar as canvas (includes bar and text)
            const barCanvas = document.createElement('canvas');
            barCanvas.width = thumbWidth;
            barCanvas.height = barHeight + textHeight;
            barCanvas.className = 'h-bar';
            const barCtx = barCanvas.getContext('2d');

            // Clear background (transparent)
            barCtx.clearRect(0, 0, thumbWidth, barHeight + textHeight);

            // Draw the H value bar extending from zero (no background, just bar and tick)
            if (hRange >= 0.0000001) {
                const hX = ((placement.hValue - this.allTimeMinH) / hRange) * thumbWidth;

                if (placement.hValue >= 0) {
                    // Positive: draw from zeroX to hX (green)
                    barCtx.fillStyle = '#4CAF50';
                    const barStart = Math.min(zeroX, hX);
                    const barEnd = Math.max(zeroX, hX);
                    barCtx.fillRect(barStart, 0, barEnd - barStart, barHeight);
                } else {
                    // Negative: draw from hX to zeroX (red)
                    barCtx.fillStyle = '#f44336';
                    const barStart = Math.min(zeroX, hX);
                    const barEnd = Math.max(zeroX, hX);
                    barCtx.fillRect(barStart, 0, barEnd - barStart, barHeight);
                }
            }

            // Draw zero tick mark
            barCtx.strokeStyle = '#333';
            barCtx.lineWidth = 1;
            barCtx.beginPath();
            barCtx.moveTo(Math.floor(zeroX) + 0.5, 0);
            barCtx.lineTo(Math.floor(zeroX) + 0.5, barHeight);
            barCtx.stroke();

            // Draw H value text below the bar
            barCtx.fillStyle = '#333';
            barCtx.font = '7px Arial';
            barCtx.textAlign = 'center';
            barCtx.fillText(placement.hValue.toFixed(3), thumbWidth / 2, barHeight + textHeight - 1);

            // Add tooltip
            canvas.title = `H = ${placement.hValue.toFixed(4)}, Col avg = ${placement.avgCol.toFixed(1)}`;

            div.appendChild(barCanvas);
            this.tetrisThumbnailsContainer.appendChild(div);
        }
    }

    /**
     * Check if two extended action lists match
     */
    _extActionsMatch(actList1, actList2) {
        if (!actList1 || !actList2) return false;
        if (actList1.length !== actList2.length) return false;
        for (let i = 0; i < actList1.length; i++) {
            if (actList1[i] !== actList2[i]) return false;
        }
        return true;
    }

    /**
     * Render a single Tetris thumbnail showing the board with ghost piece
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} obs - Current observation
     * @param {Object} placement - Placement data {extAction, blockId, rotation}
     * @param {number} cellSize - Size of each cell in pixels
     */
    _renderTetrisThumbnail(ctx, obs, placement, cellSize) {
        const boardWidth = 10;
        const boardHeight = 20;

        // Clear background (white)
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, boardWidth * cellSize, boardHeight * cellSize);

        // Draw existing blocks from observation (worldState is first 200 values)
        // Use dark gray for existing pieces
        ctx.fillStyle = '#555';
        for (let y = 0; y < boardHeight; y++) {
            for (let x = 0; x < boardWidth; x++) {
                const idx = y * boardWidth + x;
                const cellValue = obs[idx];
                if (cellValue > 0) {
                    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                }
            }
        }

        // Draw ghost piece at placement position in bright color
        const featGen = this.agent.coreAgent.featGen;
        const piece = featGen.gameState.possibleBlocks[placement.blockId].getShape(placement.rotation);

        // Use bright magenta for the new piece (stands out against dark gray existing pieces)
        ctx.fillStyle = '#FF00FF';

        for (let px = 0; px < piece.length; px++) {
            for (let py = 0; py < piece[0].length; py++) {
                if (piece[px][py] !== 0) {
                    const boardX = placement.extAction.currentX + px;
                    const boardY = placement.extAction.currentY + py;
                    if (boardX >= 0 && boardX < boardWidth && boardY >= 0 && boardY < boardHeight) {
                        ctx.fillRect(boardX * cellSize, boardY * cellSize, cellSize, cellSize);
                    }
                }
            }
        }

        // Draw hairline border around the board
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, boardWidth * cellSize - 1, boardHeight * cellSize - 1);
    }

    /**
     * Flash reward indicator (to be called by UI)
     */
    _flashRewardIndicator(type) {
        const indicator = document.getElementById('reward-indicator');
        if (indicator) {
            indicator.className = `flash-${type}`;
            setTimeout(() => indicator.className = '', 200);
        }
    }

    /**
     * Update training indicator (to be called by UI)
     */
    _updateTrainingIndicator(isTraining) {
        const indicator = document.getElementById('training-indicator');
        if (indicator) {
            indicator.textContent = isTraining ? 'Training: ON' : 'Training: OFF';
            indicator.className = isTraining ? 'training-on' : 'training-off';
        }
    }

    /**
     * Flash the training indicator to alert user that feedback requires training to be on
     */
    _flashTrainingIndicator() {
        const indicator = document.getElementById('training-indicator');
        if (!indicator) return;
        indicator.classList.add('training-off-flash');
        setTimeout(() => {
            indicator.classList.remove('training-off-flash');
        }, 300);
    }

    /**
     * Update speed indicator (to be called by UI)
     */
    _updateSpeedIndicator() {
        const indicator = document.getElementById('speed-indicator');
        if (indicator) {
            const freq = 1000 / this.stepDurationMs;
            let speedText;
            if (freq >= 100) {
                speedText = Math.round(freq).toString();
            } else if (freq >= 1) {
                speedText = freq.toFixed(1);
            } else {
                speedText = freq.toFixed(2);
            }
            indicator.textContent = `${speedText} steps/sec`;
        }
        // Sync slider thumb position
        const slider = document.getElementById('speed-slider');
        if (slider) {
            slider.value = this.stepDurationMs;
        }
    }

    /**
     * Set step duration
     */
    setStepDuration(ms) {
        this.stepDurationMs = ms;
        this._updateSpeedIndicator();
    }

    /**
     * Get current statistics
     */
    getStats() {
        return {
            ...this.agent.getStats(),
            totalEpisodes: this.totalEpisodes,
            stepsThisEp: this.stepsThisEp,
            rewardsGiven: { ...this.rewardsGiven },
            isRunning: this.running,
            isPaused: this.paused,
            isTraining: this.agent.isTraining(),
            isManualMode: this.manualControlMode
        };
    }

    /**
     * Set manual control mode
     * @param {boolean} enabled - Whether manual control is enabled
     * @returns {boolean} Whether the mode was successfully set
     */
    setManualControlMode(enabled) {
        // Tetris uses extended actions, manual control not supported
        if (this.envName === 'tetris') {
            console.warn('Manual control not supported for Tetris (uses extended actions)');
            this.manualControlMode = false;
            this._updateModeIndicator(false);
            return false;
        }

        // Manual mode and training are mutually exclusive
        // Turn off training first if needed
        if (enabled && this.agent.isTraining()) {
            this.agent.setTraining(false);
            this._updateTrainingIndicator(false);
        }

        this.manualControlMode = enabled;
        this.lastUserActI = -1;  // Reset user action
        this._updateModeIndicator(enabled);
        return true;
    }

    /**
     * Toggle manual control mode
     * @returns {boolean} New manual control state
     */
    toggleManualControlMode() {
        // Tetris uses extended actions, manual control not supported
        if (this.envName === 'tetris') {
            console.warn('Manual control not supported for Tetris (uses extended actions)');
            return false;
        }
        this.setManualControlMode(!this.manualControlMode);
        return this.manualControlMode;
    }

    /**
     * Check if manual control mode is enabled
     * @returns {boolean}
     */
    isManualControlMode() {
        return this.manualControlMode;
    }

    /**
     * Update mode indicator (to be called by UI)
     */
    _updateModeIndicator(isManual) {
        const indicator = document.getElementById('mode-indicator');
        if (indicator) {
            indicator.textContent = isManual ? 'Mode: Manual' : 'Mode: TAMER';
            indicator.className = isManual ? 'mode-manual' : 'mode-tamer';
        }
    }

    /**
     * Enable/disable debug logging for credit assignment
     * Call from console: app.setDebug(true)
     */
    setDebug(enabled) {
        this.agent.setDebug(enabled);
    }

    /**
     * Set callback for step events
     */
    onStep(callback) {
        this.onStepCallback = callback;
    }

    /**
     * Set callback for episode end events
     */
    onEpisodeEnd(callback) {
        this.onEpisodeEndCallback = callback;
    }

    /**
     * Set callback for reward events
     */
    onReward(callback) {
        this.onRewardCallback = callback;
    }
}

// Export for use as a module
export {
    // Core agents
    TamerAgent,
    SarsaLambdaAgent,
    TamerRLAgent,
    CombinationMethods,
    ImitationAgent,
    ExtActionAgentWrap,

    // Core components
    ActionSelect,
    HInfluence,

    // Feature generators
    TetrisFeatures,

    // Environments
    LoopMaze,
    MountainCar,
    CartPole,
    Acrobot,
    Tetris,
    TetrisActions,
    RobotArm
};

// Also attach to window for non-module usage
if (typeof window !== 'undefined') {
    // App
    window.TamerApp = TamerApp;

    // Agents
    window.TamerAgent = TamerAgent;
    window.SarsaLambdaAgent = SarsaLambdaAgent;
    window.TamerRLAgent = TamerRLAgent;
    window.CombinationMethods = CombinationMethods;
    window.ImitationAgent = ImitationAgent;
    window.ExtActionAgentWrap = ExtActionAgentWrap;

    // Core components
    window.ActionSelect = ActionSelect;
    window.HInfluence = HInfluence;

    // Feature generators
    window.TetrisFeatures = TetrisFeatures;

    // Environments
    window.LoopMaze = LoopMaze;
    window.MountainCar = MountainCar;
    window.CartPole = CartPole;
    window.Acrobot = Acrobot;
    window.Tetris = Tetris;
    window.TetrisActions = TetrisActions;
    window.RobotArm = RobotArm;
}
