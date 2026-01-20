/**
 * ExtActionAgentWrap - Extended Action Agent Wrapper
 * EXACT port from ExtActionAgentWrap.java in the TAMER project
 *
 * Used for domains where a sequence of multiple atomic actions are
 * presented to the agent as a single action choice.
 *
 * This class is specifically used for Tetris, where atomic actions are
 * left, right, etc. moves but the learning literature typically abstracts
 * actions to choices of piece placements.
 *
 * @author bradknox
 */

import { TamerAgent } from '../core/TamerAgent.js?v=9';

export class ExtActionAgentWrap {
    /**
     * @param {Environment} environment - The environment
     * @param {Object} options - Configuration options
     */
    constructor(environment, options = {}) {
        this.env = environment;
        this.envSpec = null;

        // Core agent (default is TamerAgent)
        this.coreAgent = null;
        this.coreAgentClass = options.coreAgentClass || TamerAgent;
        this.coreAgentOptions = options.coreAgentOptions || {};

        // Extended action tracking
        this.currExtendedAction = null;
        this.currExtActI = 0;
        this.rewThisExtAct = 0;

        // Options - EXACT from Java
        this.callCoreAgentEveryStep = options.callCoreAgentEveryStep ?? false;

        // Statistics
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;

        // Current observation (for compatibility with TamerApp)
        this.currentObs = null;
    }

    /**
     * Create inner agent - EXACT from Java createInnerAgent()
     */
    createInnerAgent() {
        if (this.coreAgent === null) {
            this.coreAgent = new this.coreAgentClass(this.env, this.coreAgentOptions);
        }
    }

    /**
     * Initialize agent - EXACT from Java agent_init()
     */
    init() {
        this.createInnerAgent();
        // envSpec is already set by the coreAgent's constructor calling env.init()
        this.envSpec = this.coreAgent.envSpec;
        return this.envSpec;
    }

    /**
     * Start episode - EXACT from Java agent_start()
     * @returns {number} First atomic action
     */
    startEpisode() {
        // Start the environment and get initial observation
        this.currentObs = this.env.start();

        // Pass observation to core agent before starting episode
        if (this.coreAgent.setCurrentObs) {
            this.coreAgent.setCurrentObs(this.currentObs);
        }

        const extendedAction = this.coreAgent.startEpisode();
        this.rewThisExtAct = 0;
        this.episodeCount++;
        this.stepsThisEp = 0;

        // Handle extended action (array of atomic actions)
        if (Array.isArray(extendedAction)) {
            this.currExtendedAction = extendedAction;
        } else {
            // Single action - wrap in array
            this.currExtendedAction = [extendedAction];
        }

        this.currExtActI = 0;
        return this.currExtendedAction[this.currExtActI];
    }

    /**
     * Take a step - EXACT from Java agent_step()
     * @param {number} time - Current time in milliseconds
     * @returns {{action: number, obs: number[], reward: number, terminal: boolean}}
     */
    step(time) {
        // Take atomic action in environment
        const atomicAction = this.currExtendedAction[this.currExtActI];
        const result = this.env.step(atomicAction);

        // Update current observation
        this.currentObs = result.obs;

        this.rewThisExtAct += result.reward;
        this.currExtActI++;
        this.stepsThisEp++;
        this.totalSteps++;

        // Check if episode ended
        if (result.terminal) {
            return {
                action: atomicAction,
                obs: result.obs,
                reward: result.reward,
                terminal: result.terminal
            };
        }

        // Check if we need a new extended action
        if (this.currExtActI >= this.currExtendedAction.length ||
            this.callCoreAgentEveryStep) {
            // Update core agent's observation before getting new extended action
            if (this.coreAgent.setCurrentObs) {
                this.coreAgent.setCurrentObs(result.obs);
            }

            // Ask core agent for new extended action
            const stepResult = this.coreAgent.step(time);

            // Get new extended action from core agent
            const extendedAction = stepResult.action;
            if (Array.isArray(extendedAction)) {
                this.currExtendedAction = extendedAction;
            } else {
                this.currExtendedAction = [extendedAction];
            }

            this.rewThisExtAct = 0;
            this.currExtActI = 0;
        }

        return {
            action: this.currExtendedAction[this.currExtActI],
            obs: result.obs,
            reward: result.reward,
            terminal: result.terminal
        };
    }

    /**
     * End episode - EXACT from Java agent_end()
     */
    endEpisode(reward) {
        this.rewThisExtAct += reward;
        console.log(`${this.episodeCount}: reward this episode`);
        this.coreAgent.endEpisode(reward);
    }

    /**
     * Process human feedback
     * @param {number} reward - The reward signal (+1 or -1)
     * @param {number} time - Time the reward was given (in milliseconds)
     */
    processHumanReward(reward, time) {
        if (this.coreAgent.processHumanReward) {
            this.coreAgent.processHumanReward(reward, time);
        }
    }

    /**
     * Toggle training mode - delegates to core agent
     */
    toggleTraining() {
        return this.coreAgent.toggleTraining();
    }

    /**
     * Check if training is enabled
     */
    isTraining() {
        return this.coreAgent.isTraining ? this.coreAgent.isTraining() : this.coreAgent.inTrainSess;
    }

    /**
     * Process key input - delegates to core agent
     */
    receiveKeyInput(key) {
        if (this.coreAgent.receiveKeyInput) {
            this.coreAgent.receiveKeyInput(key);
        }
    }

    /**
     * Get action values from core agent
     */
    getActionValues() {
        if (this.coreAgent.getActionValues) {
            return this.coreAgent.getActionValues();
        }
        return [];
    }

    /**
     * Get statistics
     */
    getStats() {
        const coreStats = this.coreAgent.getStats ? this.coreAgent.getStats() : {};
        return {
            episodeCount: this.episodeCount,
            totalSteps: this.totalSteps,
            stepsThisEp: this.stepsThisEp,
            currExtActI: this.currExtActI,
            extActLength: this.currExtendedAction ? this.currExtendedAction.length : 0,
            coreAgent: coreStats
        };
    }

    /**
     * Reset the agent
     */
    reset() {
        this.currExtendedAction = null;
        this.currExtActI = 0;
        this.rewThisExtAct = 0;
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.stepsThisEp = 0;
        this.currentObs = null;
        if (this.coreAgent && this.coreAgent.reset) {
            this.coreAgent.reset();
        }
    }
}
