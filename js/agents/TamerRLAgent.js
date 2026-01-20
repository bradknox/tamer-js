/**
 * TamerRLAgent - Combined TAMER + RL Agent
 * EXACT port from TamerRLAgent.java in the TAMER project
 *
 * An agent that combines TAMER (learning from human feedback) with
 * SARSA(λ) reinforcement learning.
 */
import { TamerAgent } from '../core/TamerAgent.js';
import { SarsaLambdaAgent } from './SarsaLambdaAgent.js';
import { HInfluence } from '../core/HInfluence.js';

// Combination method constants - EXACT from Java
export const CombinationMethods = {
    RL_ON_H_AS_R: -2,    // Use H-hat prediction as MDP reward
    TAMER_ONLY: -1,      // Only use TAMER
    RL_ONLY: 0,          // Only use RL
    REW_SHAPING: 1,      // Reward shaping with H-hat
    FEAT_ADD: 2,         // Feature addition
    Q_INIT: 3,           // Q initialization (not implemented)
    Q_AUGM: 4,           // Q augmentation
    EXTRA_ACT: 5,        // Extra action (not implemented)
    ACT_BIASING: 6,      // Action biasing
    BERNOULLI_ACT: 7,    // Bernoulli action selection (control sharing)
    STATE_POT_FCN_SHAPING: 8,  // State potential function shaping
    SA_POT_FCN_SHAPING: 9,     // State-action potential function shaping
    PROB_ACT_W_OSCILL_DAMP: 10 // Probabilistic action with oscillation damping
};

export class TamerRLAgent {
    /**
     * @param {Environment} environment - The environment
     * @param {Object} options - Configuration options
     */
    constructor(environment, options = {}) {
        this.env = environment;
        this.envSpec = environment.init();

        // Combination method - EXACT from Java defaults
        this.COMBINATION_METHOD = options.combinationMethod ?? CombinationMethods.ACT_BIASING;
        this.INITIAL_COMB_PARAM = options.combinationParam ?? 10.0;
        this.H_INFLUENCE_METHOD = options.hInfluenceMethod ?? 'annealedParam';
        this.SIMUL_LEARNING = options.simulLearning ?? false;

        // Agent options
        this.agentOptions = {
            epsilon: options.epsilon ?? 0.0,
            stepSize: options.stepSize ?? 0.001,
            basisFcnsPerDim: options.basisFcnsPerDim ?? 10,
            discountFactor: options.discountFactor ?? 0.99,
            creditDelayMs: options.creditDelayMs ?? 200,
            creditWindowMs: options.creditWindowMs ?? 600,
            creditDistType: options.creditDistType ?? 'uniform'
        };

        // Initialize sub-agents and components
        this._initComponents();

        // State tracking
        this.currentObs = null;
        this.lastObs = null;
        this.currentAction = null;
        this.lastAction = null;

        // Statistics
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;

        // Training state
        this.inTrainSess = true;
        this.tamerControl = this.COMBINATION_METHOD === CombinationMethods.TAMER_ONLY;

        // Human rewards
        this.pendingRewards = [];
    }

    /**
     * Initialize sub-agents and components - EXACT from Java agent_init()
     */
    _initComponents() {
        // Create TAMER agent
        this.tamerAgent = new TamerAgent(this.env, {
            ...this.agentOptions,
            isTopLevel: false
        });

        // Create RL agent
        this.rlAgent = new SarsaLambdaAgent(this.env, {
            epsilon: this.agentOptions.epsilon,
            stepSize: this.agentOptions.stepSize,
            basisFcnsPerDim: this.agentOptions.basisFcnsPerDim,
            discountFactor: this.agentOptions.discountFactor
        });

        // Create H-influence calculator - EXACT from Java
        const hInfStateOnlyFeats = this.COMBINATION_METHOD === CombinationMethods.BERNOULLI_ACT;

        this.hInf = new HInfluence(
            this.H_INFLUENCE_METHOD,
            this.INITIAL_COMB_PARAM * (
                this.COMBINATION_METHOD === CombinationMethods.SA_POT_FCN_SHAPING ?
                this.agentOptions.discountFactor : 1.0
            ),
            {
                stateOnly: hInfStateOnlyFeats,
                featGen: this.tamerAgent.featGen,
                stepDecayFactor: 1.0,
                epDecayFactor: 1.0,
                creditAssignParams: {
                    distClass: this.agentOptions.creditDistType,
                    creditDelayMs: this.agentOptions.creditDelayMs,
                    creditWindowMs: this.agentOptions.creditWindowMs
                }
            }
        );

        // Set constant decay factors for certain methods - EXACT from Java
        if (this.COMBINATION_METHOD !== CombinationMethods.ACT_BIASING &&
            this.COMBINATION_METHOD !== CombinationMethods.REW_SHAPING &&
            this.COMBINATION_METHOD !== CombinationMethods.BERNOULLI_ACT) {
            this.hInf.setEpDecayFactor(1.0);
            this.hInf.setStepDecayFactor(1.0);
        }

        // Set up action biasing if applicable - EXACT from Java
        if (this.COMBINATION_METHOD === CombinationMethods.ACT_BIASING ||
            this.COMBINATION_METHOD === CombinationMethods.SA_POT_FCN_SHAPING) {
            this.rlAgent.actSelector.addModelForActBias(this.tamerAgent.model, this.hInf);
        }

        // Set up Q augmentation if applicable
        if (this.COMBINATION_METHOD === CombinationMethods.Q_AUGM) {
            this.rlAgent.qAugModel = this.tamerAgent.model;
            this.rlAgent.actSelector.addModelForActBias(this.rlAgent.qAugModel, this.hInf);
        }

        // Set up RL on H as R if applicable
        if (this.COMBINATION_METHOD === CombinationMethods.RL_ON_H_AS_R) {
            this.rlAgent.actSelector.setRewModel(this.tamerAgent.model);
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
        this.pendingRewards = [];

        // Start sub-agents
        if (this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
            this.tamerAgent.startEpisode();
        }

        // Select first action using agent_step logic
        this.currentAction = this._selectAction(this.currentObs);

        // Start RL agent with the chosen action
        this.rlAgent.startEpisode();
        this.rlAgent.currentAction = this.currentAction;

        // Sync TAMER agent's action
        if (this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
            this.tamerAgent.lastAction = this.currentAction;
        }

        return this.currentAction;
    }

    /**
     * Take a step - EXACT from Java agent_step()
     * @param {number} time - Current time in milliseconds
     * @returns {{action: number, obs: number[], reward: number, terminal: boolean}}
     */
    step(time) {
        const timeInSec = time / 1000;

        // Record time step end for H influence
        this.hInf.recordTimeStepEnd(timeInSec);

        // Process pending human rewards for TAMER
        if (this.pendingRewards.length > 0 && this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
            this.tamerAgent.hLearner.processHRew(this.pendingRewards);
            this.pendingRewards = [];
        }

        // Take action in environment
        const result = this.env.step(this.currentAction);

        // Store previous state
        this.lastObs = this.currentObs;
        this.lastAction = this.currentAction;
        this.currentObs = result.obs;
        this.stepsThisEp++;
        this.totalSteps++;

        // TAMER update - EXACT from Java
        if (this.stepsThisEp > 1 && this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
            this.tamerAgent.step(time);
        }

        // H influence step update
        this.hInf.stepUpdate(this.inTrainSess, timeInSec);

        // Get manipulated reward for RL
        const manipulatedR = this._getManipulatedRew(result.reward, result.obs);

        if (!result.terminal) {
            // Select next action
            this.currentAction = this._selectAction(result.obs);

            // Record time step start for H influence
            this.hInf.recordTimeStepStart(result.obs, this.currentAction, timeInSec);

            // Sync TAMER agent
            if (this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
                this.tamerAgent.lastAction = this.currentAction;
                this.tamerAgent.hLearner.recordTimeStepStart(
                    result.obs, this.currentAction, this.tamerAgent.featGen, timeInSec
                );
            }
        }

        // RL update
        if (this.stepsThisEp > 1) {
            this.rlAgent.lastObs = this.lastObs;
            this.rlAgent.lastAction = this.lastAction;
            this.rlAgent.currentObs = this.currentObs;
            this.rlAgent.currentAction = this.currentAction;
            this.rlAgent._processPrevTimeStep(manipulatedR, result.terminal ? null : this.currentObs);
        }

        return {
            action: this.currentAction,
            obs: result.obs,
            reward: result.reward,
            terminal: result.terminal
        };
    }

    /**
     * Select action - EXACT from Java action selection logic
     */
    _selectAction(obs) {
        // RL agent selects action
        let action = this.rlAgent.actSelector.selectAction(obs, this.lastAction);

        // Check for TAMER control (Bernoulli action or tamerControl flag)
        if (this.COMBINATION_METHOD === CombinationMethods.BERNOULLI_ACT) {
            const hInfluence = this.hInf.getHInfluence(obs, null);
            if (Math.random() < hInfluence) {
                action = this.tamerAgent.selectAction(obs);
            }
        } else if (this.tamerControl) {
            action = this.tamerAgent.selectAction(obs);
        }

        return action;
    }

    /**
     * Get manipulated reward - EXACT from Java getManipulatedRew()
     */
    _getManipulatedRew(r, nextObs) {
        if (this.COMBINATION_METHOD === CombinationMethods.REW_SHAPING) {
            if (this.stepsThisEp > 1 && this.lastObs) {
                const hInfluence = this.hInf.getHInfluence(this.lastObs, this.lastAction);
                const hVal = this.tamerAgent.model.predictForAction(this.lastObs, this.lastAction);
                r += hInfluence * hVal;
            }
        } else if (this.COMBINATION_METHOD === CombinationMethods.RL_ON_H_AS_R) {
            if (this.stepsThisEp > 1 && this.lastObs) {
                r = this.tamerAgent.model.predictForAction(this.lastObs, this.lastAction);
            }
        } else if (this.COMBINATION_METHOD === CombinationMethods.STATE_POT_FCN_SHAPING) {
            if (this.lastObs && nextObs) {
                const lastStateVal = this._getStatePotential(this.lastObs);
                const nextStateVal = nextObs ? this._getStatePotential(nextObs) : 0;
                r += this.INITIAL_COMB_PARAM * (this.agentOptions.discountFactor * nextStateVal - lastStateVal);
            }
        } else if (this.COMBINATION_METHOD === CombinationMethods.SA_POT_FCN_SHAPING) {
            if (this.lastObs && this.lastAction !== null) {
                const lastSAVal = this.tamerAgent.model.predictForAction(this.lastObs, this.lastAction);
                const nextSAVal = nextObs && this.currentAction !== null ?
                    this.tamerAgent.model.predictForAction(nextObs, this.currentAction) : 0;
                r += this.INITIAL_COMB_PARAM * (this.agentOptions.discountFactor * nextSAVal - lastSAVal);
            }
        }
        return r;
    }

    /**
     * Get state potential (max Q over actions)
     */
    _getStatePotential(obs) {
        let maxVal = -Infinity;
        for (let a = 0; a < this.envSpec.numActions; a++) {
            const val = this.tamerAgent.model.predictForAction(obs, a);
            if (val > maxVal) maxVal = val;
        }
        return maxVal;
    }

    /**
     * End the current episode - EXACT from Java agent_end()
     */
    endEpisode(reward) {
        // Process any remaining rewards
        if (this.pendingRewards.length > 0) {
            this.tamerAgent.hLearner.processHRew(this.pendingRewards);
            this.pendingRewards = [];
        }

        const manipulatedR = this._getManipulatedRew(reward, null);

        this.hInf.episodeEndUpdate();

        if (this.COMBINATION_METHOD !== CombinationMethods.RL_ONLY) {
            this.tamerAgent.endEpisode();
        }

        this.rlAgent.endEpisode(manipulatedR);
    }

    /**
     * Process human feedback - EXACT from Java receiveKeyInput flow
     * @param {number} reward - The reward signal (+1 or -1)
     * @param {number} time - Time the reward was given (in milliseconds)
     */
    processHumanReward(reward, time) {
        this.pendingRewards.push({ val: reward, time: time });
    }

    /**
     * Toggle training mode
     */
    toggleTraining() {
        this.inTrainSess = !this.inTrainSess;
        this.tamerAgent.toggleTraining();
        return this.inTrainSess;
    }

    /**
     * Toggle TAMER control
     */
    toggleTamerControl() {
        if (this.COMBINATION_METHOD !== CombinationMethods.TAMER_ONLY) {
            this.tamerControl = !this.tamerControl;
        }
    }

    /**
     * Get action values from both agents
     */
    getActionValues() {
        if (!this.currentObs) return [];
        const tamerVals = this.tamerAgent.getActionValues();
        const rlVals = this.rlAgent.getActionValues();
        return {
            tamer: tamerVals,
            rl: rlVals,
            combined: tamerVals.map((t, i) => t + rlVals[i])
        };
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
            tamerControl: this.tamerControl,
            combinationMethod: this.COMBINATION_METHOD,
            tamerStats: this.tamerAgent.getStats(),
            rlStats: this.rlAgent.getStats()
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
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;
        this.pendingRewards = [];
    }
}
