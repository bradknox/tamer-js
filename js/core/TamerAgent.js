/**
 * TAMER Agent
 * EXACT port from TamerAgent.java in the TAMER project
 *
 * An agent that learns from human feedback (Training an Agent Manually
 * via Evaluative Reinforcement).
 */
import { LinearModel } from '../models/LinearModel.js';
import { RBFFeatures } from '../features/RBFFeatures.js';
import { CreditAssign } from './CreditAssign.js?v=11';
import { HLearner } from './HLearner.js?v=10';

export class TamerAgent {
    /**
     * @param {Environment} environment - The environment
     * @param {Object} options - Configuration options
     */
    constructor(environment, options = {}) {
        this.env = environment;
        this.envSpec = environment.init();

        // Agent parameters
        this.epsilon = options.epsilon ?? 0.0; // Exploration rate
        this.stepSize = options.stepSize ?? 0.05;  // EXACT from Java Params.java default

        // RBF feature parameters - EXACT from Java Params.java setPyMCParams()
        this.basisFcnsPerDim = options.basisFcnsPerDim ?? 40;
        this.relWidth = options.relWidth ?? 0.08;  // EXACT from Java default

        // Custom feature generator class (optional - uses RBFFeatures by default)
        this.FeatGenClass = options.FeatGenClass ?? null;

        // Credit assignment parameters (in milliseconds for user-facing API)
        this.creditDelayMs = options.creditDelayMs ?? 200;
        this.creditWindowMs = options.creditWindowMs ?? 600;
        this.creditDistType = options.creditDistType ?? 'uniform';
        this.extrapolateFutureRew = options.extrapolateFutureRew ?? true;  // EXACT from Java Params.java

        // Initialize components
        this._initComponents();

        // Current state
        this.currentObs = null;
        this.lastAction = null;
        this._firstStep = true;  // Track if this is the first step (no previous timestep)

        // Statistics
        this.episodeCount = 0;
        this.totalSteps = 0;

        // Pending human rewards to process
        this.pendingRewards = [];
    }

    /**
     * Initialize the learning components
     */
    _initComponents() {
        // Create feature generator
        if (this.FeatGenClass) {
            // Use custom feature generator class (e.g., TetrisFeatures)
            this.featGen = new this.FeatGenClass(
                this.envSpec.obsRanges,
                this.envSpec.numActions
            );
        } else {
            // Default: RBF features - EXACT parameters from Java FeatGen_RBFs
            this.featGen = new RBFFeatures(
                this.envSpec.obsRanges,
                this.envSpec.numActions,
                this.basisFcnsPerDim,
                this.relWidth  // From options or default 0.08
                // addBiasFeatPerAct defaults to false, we call setBiasFeatPerAct() below
            );

            // EXACT from Java TAMER setup: setNormBounds(-1, 1) and setBiasFeatPerAct(0.1)
            this.featGen.setNormBounds(-1, 1);
            this.featGen.setBiasFeatPerAct(0.1);
        }

        // Determine number of features for the model
        // For extended actions (Tetris), use getNumExtendedFeatures which returns just 46
        // For standard actions, use getNumFeatures which returns numActions * featuresPerAction
        let numModelFeatures;
        let usesExtendedActions = false;
        if (typeof this.featGen.getNumExtendedFeatures === 'function' &&
            typeof this.featGen.getPossActions === 'function') {
            // Extended actions: use smaller feature count (just 46 for Tetris)
            numModelFeatures = this.featGen.getNumExtendedFeatures();
            usesExtendedActions = true;
        } else {
            numModelFeatures = this.featGen.getNumFeatures();
        }

        // Create linear model - decayFactor=0 disables eligibility traces
        // EXACT from Java Params.java: modelAddsBiasFeat = false by default
        // Only Tetris (TetrisTamerExpHelper) sets modelAddsBiasFeat = true
        // For standard RBF environments, per-action bias is handled by setBiasFeatPerAct()
        this.model = new LinearModel(numModelFeatures, {
            stepSize: this.stepSize,
            decayFactor: 0.0,  // No eligibility traces, traces[i] = feats[i]
            useBiasWt: usesExtendedActions  // Only true for Tetris, false for standard envs
        });

        // Set feature generator for predictForAction
        this.model.setFeatGen(this.featGen);

        // Create credit assignment - EXACT parameters from Java CreditAssignParamVec
        // Note: extrapolateFutureRew defaults to true in CreditAssign (matching Java Params.java)
        // Tetris overrides this to false via options
        this.creditAssign = new CreditAssign({
            distClass: this.creditDistType,  // 'uniform', 'previousStep', or 'immediate'
            creditDelayMs: this.creditDelayMs,
            creditWindowMs: this.creditWindowMs,
            extrapolateFutureRew: this.extrapolateFutureRew,  // From options, defaults to true
            delayWtedIndivRew: false,     // Match Java default
            noUpdateWhenNoRew: false      // Match Java default
        });

        // Create human learner - EXACT from Java
        this.hLearner = new HLearner(this.model, this.creditAssign, this.featGen);
    }

    /**
     * Check if using extended actions (like Tetris piece placements)
     */
    usesExtendedActions() {
        return typeof this.featGen.getPossActions === 'function';
    }

    /**
     * Start a new episode - matches Java agent_start() which calls agent_step()
     *
     * CRITICAL: In Java, TamerAgent.agent_start() calls agent_step(), which records
     * the first timestep T0. This is essential for Tetris where ExtActionAgentWrap
     * executes the entire first extended action before calling coreAgent.step().
     * Without recording T0 here, rewards given during the first extended action
     * would have no timestep to credit!
     *
     * @returns {number|Array} First action (or extended action for Tetris)
     */
    startEpisode() {
        // If currentObs was set externally (by ExtActionAgentWrap), use it
        // Otherwise, start the environment
        if (this.currentObs === null || this.currentObs === undefined) {
            this.currentObs = this.env.start();
        }
        this.episodeCount++;

        // Process any pending rewards from the episode end pause
        // These rewards will be credited to timesteps from the previous episode
        // (if any are still in the window)
        if (this.pendingRewards.length > 0) {
            this.hLearner.processHRew(this.pendingRewards);
            this.hLearner.processSamples(performance.now(), this.creditAssign.isTraining());
        }
        this.pendingRewards = [];

        // Select first action based on starting observation
        const actionResult = this.selectAction(this.currentObs);

        // Handle extended actions (Tetris)
        if (this.usesExtendedActions()) {
            this.lastAction = actionResult.action;
            this.lastActionFeats = actionResult.feats;
        } else {
            this.lastAction = actionResult;
        }

        // CRITICAL: Record the first timestep T0 - EXACT from Java agent_start() -> agent_step()
        // In Java, agent_start() calls agent_step() which records the timestep.
        // This is essential for extended actions (Tetris) where the first extended action
        // executes entirely before coreAgent.step() is called. Without T0 being recorded
        // here, rewards during the first extended action would have no timestep to credit.
        const timeInSec = performance.now() / 1000;
        if (this.usesExtendedActions()) {
            this.hLearner.recordTimeStepWithFeats(this.lastActionFeats, timeInSec, this.currentObs, this.lastAction);
        } else {
            this.hLearner.recordTimeStepStart(this.currentObs, this.lastAction, timeInSec);
        }

        // First timestep is now recorded, so step() should end it on first call
        this._firstStep = false;

        return this.lastAction;
    }

    /**
     * Take a step - matches Java stepAgentEnv() + agent_step() flow
     *
     * CRITICAL: Java timing for credit assignment
     *
     * In Java stepAgentEnv call 2 (first step, timeStep==1):
     * 1. agent_start(S0) is called
     *    - recordTimeStepEnd (nothing to end on first step)
     *    - process rewards (nothing to credit on first step)
     *    - select A0 based on S0
     *    - recordTimeStepStart(S0, A0) - T0 STARTS
     *    - return A0
     * 2. env_step(A0) produces S1
     * 3. render S1 - T0 is ONGOING!
     *
     * In Java stepAgentEnv call 3 (second step):
     * 1. agent_step(S1) is called
     *    - recordTimeStepEnd - T0 ENDS
     *    - process rewards (credited to T0)
     *    - select A1 based on S1
     *    - recordTimeStepStart(S1, A1) - T1 STARTS
     *    - return A1
     * 2. env_step(A1) produces S2
     * 3. render S2 - T1 is ONGOING!
     *
     * Key insight: When S1 is rendered, T0=(S0,A0) is ongoing.
     * T0 is for action A0 which PRODUCED S1. User feedback credits T0 correctly.
     *
     * @param {number} time - Current time in milliseconds
     * @returns {{action: number|Array, obs: number[], reward: number, terminal: boolean}}
     */
    step(time) {
        const timeInSec = time / 1000;

        // Save the observation BEFORE any changes (for timestep recording)
        // This is lastObservation in Java - the obs used to select lastAction
        const obsBeforeStep = this.currentObs;

        // 1. End previous timestep (if any) - EXACT from Java agent_step line 137
        // On first step, there's nothing to end (matches Java)
        if (!this._firstStep) {
            this.hLearner.recordTimeStepEnd(timeInSec);
        }

        // 2. Process pending human rewards - Java: processPrevTimeStep
        if (this.pendingRewards.length > 0) {
            this.hLearner.processHRew(this.pendingRewards);
            this.pendingRewards = [];
        }

        // 3. Process samples and update model - EXACT from Java agent_step line 147
        // duringStepTransition=true because we're between recordTimeStepEnd and recordTimeStepStart
        // This affects when samples are considered "finished" for previousStep/immediate modes
        this.hLearner.processSamples(time, this.creditAssign.isTraining(), true);

        // 4. For extended actions: select action FIRST, then record timestep
        // EXACT from Java agent_step: selectAction comes BEFORE recordTimeStepStart
        // This ensures the timestep is recorded with features for (currentObs, newAction)
        if (this.usesExtendedActions()) {
            // Select action based on current observation (which is the result of the previous extended action)
            const actionResult = this._selectExtendedAction(this.currentObs);
            this.lastAction = actionResult.action;
            this.lastActionFeats = actionResult.feats;

            // Now record timestep with the CORRECT features for (currentObs, selectedAction)
            this.hLearner.recordTimeStepWithFeats(this.lastActionFeats, timeInSec, this.currentObs, this.lastAction);
        } else {
            // For non-extended: record timestep with (obsBeforeStep, lastAction)
            // obsBeforeStep is what lastAction was selected based on (from previous step)
            this.hLearner.recordTimeStepStart(obsBeforeStep, this.lastAction, timeInSec);
        }

        // Clear first step flag
        this._firstStep = false;

        // 5. NOW execute the action - EXACT from Java stepAgentEnv env_step
        // The timestep we just recorded is ONGOING during this execution
        let result;
        if (this.usesExtendedActions()) {
            // For extended actions, action was already selected above
            // Just return the current state - ExtActionAgentWrap will execute the action
            result = {
                obs: this.currentObs,
                reward: 0,
                terminal: false
            };
        } else {
            result = this.env.step(this.lastAction);
            this.currentObs = result.obs;
        }

        this.totalSteps++;

        // 6. Select next action for the NEXT step (non-extended only)
        // For extended actions, action was already selected in step 4
        if (!result.terminal && !this.usesExtendedActions()) {
            this.lastAction = this.selectAction(this.currentObs);
        }

        // The timestep we recorded is STILL ONGOING when this returns
        // It will be ended at the START of the next step() call
        return {
            action: this.lastAction,
            obs: result.obs,
            reward: result.reward,
            terminal: result.terminal
        };
    }

    /**
     * End the current episode - matches Java agent_end()
     */
    endEpisode() {
        // Process any remaining rewards
        if (this.pendingRewards.length > 0) {
            this.hLearner.processHRew(this.pendingRewards);
            this.pendingRewards = [];
        }

        // Record the final timestep end
        const time = performance.now() / 1000;
        this.creditAssign.recordTimeStepEnd(time);

        // Process any remaining samples
        this.hLearner.processSamples(performance.now(), this.creditAssign.isTraining());

        // NOTE: Do NOT call clearHistory() here - EXACT from Java TamerAgent.java line 114
        // which is commented out. This allows rewards given during the post-episode pause
        // to still credit the final actions of the episode.

        // Reset currentObs so next startEpisode() will call env.start()
        // to sample a new start state
        this.currentObs = null;
    }

    /**
     * Select an action based on predicted human reward
     * @param {number[]} obs - Current observation
     * @returns {number|{action: Array, feats: Float64Array}} Selected action
     */
    selectAction(obs) {
        // Extended actions (Tetris)
        if (this.usesExtendedActions()) {
            return this._selectExtendedAction(obs);
        }

        // Standard action selection
        // Epsilon-greedy exploration
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.envSpec.numActions);
        }

        // Get predicted human rewards for all actions
        const predictions = this.hLearner.predictAllActions(obs, this.envSpec.numActions);

        // Select action with highest predicted reward (with tie-breaking)
        let bestAction = 0;
        let bestValue = predictions[0];
        const ties = [0];

        for (let a = 1; a < this.envSpec.numActions; a++) {
            if (predictions[a] > bestValue) {
                bestValue = predictions[a];
                bestAction = a;
                ties.length = 0;
                ties.push(a);
            } else if (predictions[a] === bestValue) {
                ties.push(a);
            }
        }

        // Random tie-breaking
        if (ties.length > 1) {
            bestAction = ties[Math.floor(Math.random() * ties.length)];
        }

        return bestAction;
    }

    /**
     * Select an extended action (for Tetris piece placements)
     * Uses tree search to find all possible placements, then selects the best one
     * @param {number[]} obs - Current observation
     * @returns {{action: Array, feats: Float64Array}} Selected extended action and its features
     */
    _selectExtendedAction(obs) {
        // Get all possible extended actions from feature generator
        const extendedActions = this.featGen.getPossActions(obs);

        if (extendedActions.length === 0) {
            console.error('No extended actions available');
            return { action: [4], feats: new Float64Array(this.featGen.getNumFeatures()) };
        }

        // Epsilon-greedy exploration
        if (Math.random() < this.epsilon) {
            const randomIdx = Math.floor(Math.random() * extendedActions.length);
            const feats = this.featGen.getSAFeats(obs, extendedActions[randomIdx].actList);
            return { action: extendedActions[randomIdx].actList, feats: feats };
        }

        // Get predicted rewards for all extended actions
        const predictions = this.hLearner.predictExtendedActions(obs, extendedActions);

        // Select action with highest predicted reward (with tie-breaking)
        let bestIdx = 0;
        let bestValue = predictions[0];
        const ties = [0];

        for (let i = 1; i < predictions.length; i++) {
            if (predictions[i] > bestValue) {
                bestValue = predictions[i];
                bestIdx = i;
                ties.length = 0;
                ties.push(i);
            } else if (predictions[i] === bestValue) {
                ties.push(i);
            }
        }

        // Random tie-breaking
        if (ties.length > 1) {
            bestIdx = ties[Math.floor(Math.random() * ties.length)];
        }

        // Compute features for the selected action
        const bestFeats = this.featGen.getSAFeats(obs, extendedActions[bestIdx].actList);

        return { action: extendedActions[bestIdx].actList, feats: bestFeats };
    }

    /**
     * Process human feedback - matches Java receiveKeyInput flow
     * @param {number} reward - The reward signal (+1 or -1)
     * @param {number} time - Time the reward was given (in milliseconds)
     */
    processHumanReward(reward, time) {
        // Queue reward for processing during next step
        this.pendingRewards.push({ val: reward, time: time });
    }

    /**
     * Process a manual step - for when user controls the agent but TAMER still learns
     * This allows learning from human-chosen actions while in manual control mode
     * @param {number} action - The user-chosen action
     * @param {number[]} newObs - The new observation after taking the action
     * @param {number} time - Current time in milliseconds
     */
    processManualStep(action, newObs, time) {
        const timeInSec = time / 1000;

        // Save the observation BEFORE any changes
        const obsBeforeStep = this.currentObs;

        // End previous timestep (if any)
        if (!this._firstStep) {
            this.hLearner.recordTimeStepEnd(timeInSec);
        }

        // Process pending human rewards
        if (this.pendingRewards.length > 0) {
            this.hLearner.processHRew(this.pendingRewards);
            this.pendingRewards = [];
        }

        // Process samples and update model
        this.hLearner.processSamples(time, this.creditAssign.isTraining(), true);

        // Record timestep with the user-chosen action
        this.hLearner.recordTimeStepStart(obsBeforeStep, action, timeInSec);

        // Clear first step flag
        this._firstStep = false;

        // Update internal state
        this.currentObs = newObs;
        this.lastAction = action;
        this.totalSteps++;
    }

    /**
     * Get predicted human reward values for all actions at current state
     * For extended actions (Tetris), returns values for top placements
     * @returns {number[]}
     */
    getActionValues() {
        if (!this.currentObs) return [];

        if (this.usesExtendedActions()) {
            // For extended actions, return predictions for available placements
            const extendedActions = this.featGen.getPossActions(this.currentObs);
            return this.hLearner.predictExtendedActions(this.currentObs, extendedActions);
        }

        return this.hLearner.predictAllActions(this.currentObs, this.envSpec.numActions);
    }

    /**
     * Get number of actions (or extended actions)
     */
    getNumActions() {
        if (this.usesExtendedActions() && this.currentObs) {
            const extendedActions = this.featGen.getPossActions(this.currentObs);
            return extendedActions.length;
        }
        return this.envSpec.numActions;
    }

    /**
     * Set current observation (used by ExtActionAgentWrap)
     */
    setCurrentObs(obs) {
        this.currentObs = obs;
    }

    /**
     * Toggle training mode
     * @returns {boolean} New training state
     */
    toggleTraining() {
        return this.hLearner.toggleTraining();
    }

    /**
     * Check if training is enabled
     * @returns {boolean}
     */
    isTraining() {
        return this.hLearner.isTraining();
    }

    /**
     * Get agent statistics
     */
    getStats() {
        const hStats = this.hLearner.getStats();
        return {
            episodeCount: this.episodeCount,
            totalSteps: this.totalSteps,
            ...hStats
        };
    }

    /**
     * Set exploration rate
     */
    setEpsilon(epsilon) {
        this.epsilon = epsilon;
    }

    /**
     * Set learning rate
     */
    setStepSize(stepSize) {
        this.stepSize = stepSize;
        this.model.setStepSize(stepSize);
    }

    /**
     * Reset the agent
     */
    reset() {
        this._initComponents();
        this.currentObs = null;
        this.lastAction = null;
        this._firstStep = true;
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.pendingRewards = [];
    }

    /**
     * Enable/disable debug logging for credit assignment
     */
    setDebug(enabled) {
        this.creditAssign.setDebug(enabled);
    }
}
