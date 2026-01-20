/**
 * Human Reward Learner (HLearner)
 * EXACT port from HLearner.java in the TAMER project
 *
 * Interfaces between the agent, credit assignment, and regression model
 * to learn the human reward model.
 *
 * VERSION: 3 - Added predictExtendedActions for Tetris, fixed addInstancesWReplacement
 */
export class HLearner {
    /**
     * @param {LinearModel} model - The regression model
     * @param {CreditAssign} creditAssign - The credit assignment module
     * @param {RBFFeatures} featGen - Feature generator
     */
    constructor(model, creditAssign, featGen) {
        this.model = model;
        this.credA = creditAssign;  // Named credA to match Java
        this.featGen = featGen;

        // Statistics
        this.totalSamplesProcessed = 0;
        this.totalRewardsReceived = 0;

        // Log to verify this version is loaded
        console.log('HLearner v2 loaded - has predictExtendedActions:', typeof this.predictExtendedActions === 'function');
    }

    /**
     * Get the model - EXACT from Java getModel()
     */
    getModel() {
        return this.model;
    }

    /**
     * Reset the learner - EXACT from Java reset()
     */
    reset() {
        this.model.reset();
        this.clearHistory();
        this.totalSamplesProcessed = 0;
        this.totalRewardsReceived = 0;
    }

    /**
     * Record a timestep - EXACT from Java recordTimeStep()
     * Each time step will have a endTime, which is the next step's startTime
     * @param {number[]} obs - Observation
     * @param {number} action - Action taken
     * @param {number} startTimeMs - Time in milliseconds
     */
    recordTimeStep(obs, action, startTimeMs) {
        const startTime = startTimeMs / 1000;  // Convert ms to seconds for Java compatibility
        this.recordTimeStepEnd(startTime);
        this.recordTimeStepStart(obs, action, startTime);
    }

    /**
     * Record timestep start - EXACT from Java recordTimeStepStart()
     * @param {number[]} obs - Observation
     * @param {number} action - Action
     * @param {number} startTime - Time in seconds
     */
    recordTimeStepStart(obs, action, startTime) {
        const feats = this.featGen.getStateActionFeatures(obs, action);
        this.credA.recordTimeStepStart(feats, startTime, obs, action);
    }

    /**
     * Record timestep end - EXACT from Java recordTimeStepEnd()
     * @param {number} endTime - Time in seconds
     */
    recordTimeStepEnd(endTime) {
        this.credA.recordTimeStepEnd(endTime);
    }

    /**
     * Process samples and update model - EXACT from Java processSamples()
     * @param {number} currTimeMs - Current time in milliseconds
     * @param {boolean} inTrainSess - Whether in training session
     * @param {boolean} duringStepTransition - True if called during agent_step (affects previousStep/immediate finish bounds)
     * @returns {Array} Samples that were processed
     */
    processSamples(currTimeMs, inTrainSess, duringStepTransition = false) {
        const currTime = currTimeMs / 1000;  // Convert ms to seconds
        const samples = this.credA.processSamplesAndRemoveFinished(currTime, inTrainSess, duringStepTransition);

        if (samples.length > 0) {
            this._addSamplesAndBuild(samples);
            this.totalSamplesProcessed += samples.length;
        }

        return samples;
    }

    /**
     * Clear history - EXACT from Java clearHistory()
     */
    clearHistory() {
        this.credA.clearHistory();
    }

    /**
     * Process human rewards - EXACT from Java processHRew()
     * @param {Array<{val: number, time: number}>} hRewThisStep - Human rewards with values and times
     */
    processHRew(hRewThisStep) {
        for (const hRew of hRewThisStep) {
            const timeInSeconds = hRew.time / 1000;  // Convert ms to seconds
            this.credA.processNewHReward(hRew.val, timeInSeconds);
            this.totalRewardsReceived++;
        }
    }

    /**
     * Process a single human reward (convenience method)
     * @param {number} reward - The reward value
     * @param {number} timeMs - Time in milliseconds
     */
    processReward(reward, timeMs) {
        this.processHRew([{ val: reward, time: timeMs }]);
    }

    /**
     * Add samples and build model - EXACT from Java addSamplesAndBuild()
     *
     * Uses addInstancesWReplacement which properly handles:
     * - Complete samples (usedCredit > 0.99999): permanently added to base model
     * - Incomplete samples (usedCredit <= 0.99999): temporarily added for predictions
     *
     * This is CRITICAL for credit assignment extrapolation to work correctly.
     */
    _addSamplesAndBuild(samples) {
        this.model.addInstancesWReplacement(samples);
        this.model.buildModel();  // No-op for IncGDLinearModel but matches Java
    }

    /**
     * Predict human reward for a state-action pair
     * @param {number[]} obs - Observation
     * @param {number} action - Action
     * @returns {number} Predicted human reward
     */
    predict(obs, action) {
        const features = this.featGen.getStateActionFeatures(obs, action);
        return this.model.predict(features);
    }

    /**
     * Get predicted rewards for all actions at a given state
     * @param {number[]} obs - Observation
     * @param {number} numActions - Number of actions
     * @returns {number[]} Predicted rewards for each action
     */
    predictAllActions(obs, numActions) {
        const predictions = [];
        for (let a = 0; a < numActions; a++) {
            predictions.push(this.predict(obs, a));
        }
        return predictions;
    }

    /**
     * Predict human reward for extended actions (like Tetris piece placements)
     * Uses getSAFeats instead of getStateActionFeatures for state-change features
     * @param {number[]} obs - Observation
     * @param {Array} extendedActions - List of ExtendedTetrisAction objects
     * @returns {number[]} Predicted rewards for each extended action
     */
    predictExtendedActions(obs, extendedActions) {
        const predictions = [];
        for (const extAct of extendedActions) {
            const feats = this.featGen.getSAFeats(obs, extAct.actList);
            predictions.push(this.model.predict(feats));
        }
        return predictions;
    }

    /**
     * Record timestep with pre-computed features (for extended actions)
     * @param {Float64Array} feats - Pre-computed features
     * @param {number} startTime - Time in seconds
     * @param {number[]} obs - Observation
     * @param {Array} action - Action (can be extended action)
     */
    recordTimeStepWithFeats(feats, startTime, obs, action) {
        this.credA.recordTimeStepStart(feats, startTime, obs, action);
    }

    /**
     * Toggle training session
     * @returns {boolean} New training state
     */
    toggleTraining() {
        return this.credA.toggleTraining();
    }

    /**
     * Check if currently training
     */
    isTraining() {
        return this.credA.isTraining();
    }

    /**
     * Get statistics
     */
    getStats() {
        return {
            totalSamples: this.totalSamplesProcessed,
            totalRewards: this.totalRewardsReceived,
            modelUpdates: this.model.numUpdates
        };
    }
}
