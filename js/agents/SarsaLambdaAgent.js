/**
 * SarsaLambdaAgent - SARSA(λ) Reinforcement Learning Agent
 * EXACT port from SarsaLambdaAgent.java in the TAMER project
 *
 * Implements the SARSA(λ) algorithm with eligibility traces.
 */
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { ActionSelect } from '../core/ActionSelect.js';

export class SarsaLambdaAgent {
    /**
     * @param {Environment} environment - The environment
     * @param {Object} options - Configuration options
     */
    constructor(environment, options = {}) {
        this.env = environment;
        this.envSpec = environment.init();

        // Agent parameters - EXACT from Java
        this.epsilon = options.epsilon ?? 0.1;
        this.stepSize = options.stepSize ?? 0.01;
        this.discountFactor = options.discountFactor ?? 0.99;
        this.traceDecayFactor = options.traceDecayFactor ?? 0.9; // λ for eligibility traces

        // RBF feature parameters
        this.basisFcnsPerDim = options.basisFcnsPerDim ?? 10;

        // Initialize components
        this._initComponents();

        // Current state
        this.currentObs = null;
        this.lastObs = null;
        this.lastAction = null;
        this.currentAction = null;

        // Statistics
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;

        // Training state
        this.isTraining = true;
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

        // Create linear model with eligibility traces
        this.model = new LinearModel(this.featGen.getNumFeatures(), {
            stepSize: this.stepSize,
            decayFactor: this.traceDecayFactor
        });

        // Set feature generator for predictForAction
        this.model.setFeatGen(this.featGen);

        // Create action selector - EXACT from Java
        this.actSelector = new ActionSelect(
            this.model,
            'e-greedy',
            {
                epsilon: this.epsilon,
                epsilonAnnealRate: 0.9995
            },
            this.envSpec.numActions
        );
    }

    /**
     * Start a new episode - EXACT from Java agent_start()
     * @returns {number} First action
     */
    startEpisode() {
        this.currentObs = this.env.start();
        this.episodeCount++;
        this.stepsThisEp = 0;

        // Reset eligibility traces at episode start
        this.model.resetTraces();

        // Select first action
        this.currentAction = this.actSelector.selectAction(this.currentObs, null);
        this.lastObs = null;
        this.lastAction = null;

        return this.currentAction;
    }

    /**
     * Take a step - EXACT from Java agent_step()
     * @param {number} time - Current time in milliseconds
     * @returns {{action: number, obs: number[], reward: number, terminal: boolean}}
     */
    step(time) {
        // Take action in environment
        const result = this.env.step(this.currentAction);

        // Store for SARSA update
        const prevObs = this.currentObs;
        const prevAction = this.currentAction;

        this.currentObs = result.obs;
        this.stepsThisEp++;
        this.totalSteps++;

        // Select next action
        if (!result.terminal) {
            this.currentAction = this.actSelector.selectAction(this.currentObs, prevAction);
        }

        // Process previous time step (SARSA update)
        if (this.stepsThisEp > 1 && this.isTraining) {
            this._processPrevTimeStep(result.reward, result.terminal ? null : this.currentObs);
        }

        // Store for next iteration
        this.lastObs = prevObs;
        this.lastAction = prevAction;

        return {
            action: this.currentAction,
            obs: result.obs,
            reward: result.reward,
            terminal: result.terminal
        };
    }

    /**
     * Process previous time step with SARSA update - EXACT from Java processPrevTimeStep()
     */
    _processPrevTimeStep(reward, nextObs) {
        if (this.lastObs === null) return;

        // Get current state-action value (for non-terminal states)
        let thisSAVal = 0.0;
        if (nextObs !== null) {
            thisSAVal = this.model.predictForAction(nextObs, this.currentAction);
        }

        // Get features of previous time step
        const lastStepFeats = this.featGen.getStateActionFeatures(this.lastObs, this.lastAction);

        // SARSA target: r + γ * Q(s', a')
        const target = reward + (this.discountFactor * thisSAVal);

        // Update model with sample
        this.model.addInstance({
            feats: lastStepFeats,
            label: target,
            weight: 1.0
        });
    }

    /**
     * End the current episode - EXACT from Java agent_end()
     */
    endEpisode(reward) {
        // Process final transition (terminal state has value 0)
        if (this.stepsThisEp > 0 && this.isTraining) {
            this._processPrevTimeStep(reward, null);
        }

        // Anneal exploration
        this.actSelector.anneal();
    }

    /**
     * Select an action based on Q-values
     * @param {number[]} obs - Current observation
     * @returns {number} Selected action
     */
    selectAction(obs) {
        return this.actSelector.selectAction(obs, this.lastAction);
    }

    /**
     * Get Q-values for all actions at current state
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
     * Get Q-value for last state-action
     */
    getValForLastStep() {
        if (!this.lastObs || this.lastAction === null) return 0;
        return this.model.predictForAction(this.lastObs, this.lastAction);
    }

    /**
     * Toggle training mode
     * @returns {boolean} New training state
     */
    toggleTraining() {
        this.isTraining = !this.isTraining;
        return this.isTraining;
    }

    /**
     * Set exploration rate
     */
    setEpsilon(epsilon) {
        this.epsilon = epsilon;
        this.actSelector.selectionParams.epsilon = epsilon;
    }

    /**
     * Set discount factor
     */
    setDiscountFactor(df) {
        this.discountFactor = df;
        this.actSelector.setDiscountParam(ActionSelect.discFactorToParam(df));
    }

    /**
     * Get agent statistics
     */
    getStats() {
        return {
            episodeCount: this.episodeCount,
            totalSteps: this.totalSteps,
            stepsThisEp: this.stepsThisEp,
            epsilon: this.actSelector.selectionParams.epsilon,
            isTraining: this.isTraining
        };
    }

    /**
     * Reset the agent
     */
    reset() {
        this._initComponents();
        this.currentObs = null;
        this.lastObs = null;
        this.lastAction = null;
        this.currentAction = null;
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;
    }
}
