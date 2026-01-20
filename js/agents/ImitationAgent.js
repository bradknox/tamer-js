/**
 * ImitationAgent - Learning from Demonstration Agent
 * EXACT port from ImitationAgent.java in the TAMER project
 *
 * Implements learning from demonstration by policy regression.
 * A human controls the agent and can switch off training at any time
 * to see the learned policy.
 */
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { ActionSelect } from '../core/ActionSelect.js';

export class ImitationAgent {
    /**
     * @param {Environment} environment - The environment
     * @param {Object} options - Configuration options
     */
    constructor(environment, options = {}) {
        this.env = environment;
        this.envSpec = environment.init();
        this.envName = environment.getName ? environment.getName() : 'unknown';

        // Agent parameters
        this.stepSize = options.stepSize ?? 0.01;
        this.basisFcnsPerDim = options.basisFcnsPerDim ?? 10;

        // Control mode - EXACT from Java
        this.controlOnly = options.controlOnly ?? false;
        this.okayToHang = options.okayToHang ?? false; // In browser, we don't hang

        // Initialize components
        this._initComponents();

        // Current state
        this.currentObs = null;
        this.lastObs = null;
        this.currentAction = null;
        this.lastAction = null;

        // User action tracking - EXACT from Java
        this.lastUserActI = -1;

        // Statistics
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;

        // Training state
        this.inTrainSess = true;
        this.allowUserToggledTraining = !this.controlOnly;
    }

    /**
     * Initialize the learning components - EXACT from Java agent_init()
     */
    _initComponents() {
        // Create feature generator - EXACT parameters from Java
        this.featGen = new RBFFeatures(
            this.envSpec.obsRanges,
            this.envSpec.numActions,
            this.basisFcnsPerDim,
            0.08  // relWidth - EXACT from Java
        );

        // Set normalization bounds - EXACT from Java
        this.featGen.setNormBounds(-1, 1);
        this.featGen.setBiasFeatPerAct(0.1);

        // Create linear model for policy regression
        this.model = new LinearModel(this.featGen.getNumFeatures(), {
            stepSize: this.stepSize,
            decayFactor: 0.0  // No eligibility traces for imitation
        });

        // Set feature generator for predictForAction
        this.model.setFeatGen(this.featGen);

        // Create action selector - EXACT from Java
        this.actSelector = new ActionSelect(
            this.model,
            'greedy',
            {},
            this.envSpec.numActions
        );
    }

    /**
     * Set control-only mode before start - EXACT from Java setControlOnlyBeforeStart()
     */
    setControlOnlyBeforeStart(controlOnly) {
        this.controlOnly = controlOnly;
        if (controlOnly) {
            this.inTrainSess = true;
        }
    }

    /**
     * Check if in control-only mode
     */
    isControlOnly() {
        return this.controlOnly;
    }

    /**
     * Process key input for human control - EXACT from Java receiveKeyInput()
     * This maps keyboard keys to actions based on environment
     */
    receiveKeyInput(key) {
        // Toggle training with space bar
        if (key === ' ' && !this.controlOnly && this.allowUserToggledTraining) {
            this.toggleTraining();
            return;
        }

        // Environment-specific key mappings - EXACT from Java
        if (this.envName === 'LoopMaze' || this.envName === 'Loop-Maze' ||
            this.envName === 'Puddle-World' || this.envName === 'Grid-World') {
            // 0=right, 1=left, 2=down, 3=up
            if (key === 'j') this.lastUserActI = 1;      // left
            else if (key === 'k') this.lastUserActI = 2; // down
            else if (key === 'l') this.lastUserActI = 0; // right
            else if (key === 'i') this.lastUserActI = 3; // up
        } else if (this.envName === 'CartPole') {
            // 0=left, 1=right
            if (key === 'j') this.lastUserActI = 0;      // accelerate left
            else if (key === 'l') this.lastUserActI = 1; // accelerate right
        } else if (this.envName === 'Mountain-Car' || this.envName === 'MountainCar') {
            // 0=left, 1=neutral, 2=right
            if (key === 'j') this.lastUserActI = 0;      // accelerate left
            else if (key === 'k') this.lastUserActI = 1; // don't accelerate
            else if (key === 'l') this.lastUserActI = 2; // accelerate right
        } else if (this.envName === 'Acrobot') {
            // 0=left, 1=none, 2=right
            if (key === 'j') this.lastUserActI = 0;      // torque left
            else if (key === 'k') this.lastUserActI = 1; // no torque
            else if (key === 'l') this.lastUserActI = 2; // torque right
        } else if (this.envSpec.numActions === 4) {
            // Default 4-action mapping
            if (key === 'j') this.lastUserActI = 1;      // left
            else if (key === 'k') this.lastUserActI = 2; // down
            else if (key === 'l') this.lastUserActI = 0; // right
            else if (key === 'i') this.lastUserActI = 3; // up
        } else {
            // Generic two-action mapping
            if (key === '/') this.lastUserActI = this.envSpec.numActions - 1;
            else if (key === 'z') this.lastUserActI = 0;
        }
    }

    /**
     * Start a new episode - EXACT from Java agent_start()
     * @returns {number} First action
     */
    startEpisode() {
        this.currentObs = this.env.start();
        this.episodeCount++;
        this.stepsThisEp = 0;
        this.lastUserActI = -1;

        // Select first action (agent-chosen or user-controlled)
        if (this.inTrainSess && this.lastUserActI !== -1) {
            this.currentAction = this.lastUserActI;
        } else {
            this.currentAction = this.actSelector.selectAction(this.currentObs, null);
        }

        this.lastObs = null;
        this.lastAction = null;

        return this.currentAction;
    }

    /**
     * Take a step - EXACT from Java agent_step()
     * @param {number} time - Current time in milliseconds
     * @param {number} userAction - User-provided action (optional, for web interface)
     * @returns {{action: number, obs: number[], reward: number, terminal: boolean}}
     */
    step(time, userAction = null) {
        // Process user action if provided
        if (userAction !== null) {
            this.lastUserActI = userAction;
        }

        // Store previous state
        this.lastObs = this.currentObs;
        this.lastAction = this.currentAction;

        // Process previous time step (learning from demonstration)
        if (this.stepsThisEp > 0) {
            this._processPrevTimeStep();
        }

        // Take action in environment
        const result = this.env.step(this.currentAction);
        this.currentObs = result.obs;
        this.stepsThisEp++;
        this.totalSteps++;

        // Select next action
        if (!result.terminal) {
            if (this.inTrainSess && this.lastUserActI !== -1) {
                // Use user-provided action
                this.currentAction = this.lastUserActI;
                this.lastUserActI = -1; // Reset for next step
            } else {
                // Use learned policy
                this.currentAction = this.actSelector.selectAction(result.obs, this.lastAction);
            }
        }

        return {
            action: this.currentAction,
            obs: result.obs,
            reward: result.reward,
            terminal: result.terminal
        };
    }

    /**
     * Process previous time step for learning - EXACT from Java processPrevTimeStep()
     */
    _processPrevTimeStep() {
        if (!this.inTrainSess || this.lastUserActI === -1 || !this.lastObs) {
            return;
        }

        // For each action, create a sample with label 1 for demonstrated action, 0 otherwise
        // This is policy regression - EXACT from Java
        for (let a = 0; a < this.envSpec.numActions; a++) {
            const label = (a === this.lastAction) ? 1.0 : 0.0;
            const feats = this.featGen.getStateActionFeatures(this.lastObs, a);

            this.model.addInstance({
                feats: feats,
                label: label,
                weight: 1.0
            });
        }

        // Build model after adding samples
        this.model.buildModel();
    }

    /**
     * End the current episode - EXACT from Java agent_end()
     */
    endEpisode(reward) {
        // Process final time step
        if (this.stepsThisEp > 0) {
            this._processPrevTimeStep();
        }
        this.lastUserActI = -1;
    }

    /**
     * Toggle training mode
     * @returns {boolean} New training state
     */
    toggleTraining() {
        if (this.allowUserToggledTraining) {
            this.inTrainSess = !this.inTrainSess;
        }
        return this.inTrainSess;
    }

    /**
     * Check if training is enabled
     * @returns {boolean}
     */
    isTraining() {
        return this.inTrainSess;
    }

    /**
     * Get action values (policy probabilities)
     * @returns {number[]}
     */
    getActionValues() {
        if (!this.currentObs) return [];
        const values = [];
        for (let a = 0; a < this.envSpec.numActions; a++) {
            values.push(this.model.predictForAction(this.currentObs, a));
        }
        return values;
    }

    /**
     * Get statistics
     */
    getStats() {
        return {
            episodeCount: this.episodeCount,
            totalSteps: this.totalSteps,
            stepsThisEp: this.stepsThisEp,
            inTrainSess: this.inTrainSess,
            controlOnly: this.controlOnly
        };
    }

    /**
     * Reset the agent
     */
    reset() {
        this._initComponents();
        this.currentObs = null;
        this.lastObs = null;
        this.currentAction = null;
        this.lastAction = null;
        this.lastUserActI = -1;
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;
    }
}
