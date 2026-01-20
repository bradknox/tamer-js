/**
 * Credit Assignment for TAMER
 * EXACT port from CreditAssign.java in the TAMER project
 *
 * Distributes human feedback rewards across past state-action pairs
 * based on temporal credit assignment.
 */
export class CreditAssign {
    /**
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        // Distribution type: 'uniform', 'previousStep', or 'immediate'
        this.distClass = options.distClass ?? 'uniform';

        // Credit window parameters (in seconds, matching Java)
        // windowStart = creditDelay, windowEnd = creditDelay + windowSize
        this.windowStart = (options.creditDelayMs ?? 200) / 1000;  // Convert ms to seconds
        this.windowEnd = this.windowStart + (options.creditWindowMs ?? 600) / 1000;

        // Extrapolation settings - EXACT from Java Params.java line 71
        this.EXTRAPOLATE_FUTURE_REW = options.extrapolateFutureRew ?? true;
        this.MIN_USED_CRED_FOR_EXTRAP = 0.5;
        this.SAMPLE_CUMUL_CRED_MIN = 0.9;
        this.APPROX_ONE = 0.99999;

        // delayWtedIndivRew mode (legacy, usually false)
        this.delayWtedIndivRew = options.delayWtedIndivRew ?? false;
        this.noUpdateWhenNoRew = options.noUpdateWhenNoRew ?? false;

        if (this.delayWtedIndivRew) {
            this.EXTRAPOLATE_FUTURE_REW = false;
            this.MIN_USED_CRED_FOR_EXTRAP = 0.0;
        }

        // Unique ID counter for samples
        this.UNIQUE_START = options.uniqueStart ?? 0;
        this.totalTimeSteps = 0;

        // History tracking - EXACT from Java
        this.timeStepsInWindow = [];  // TimeStepForCred objects
        this.activeSamples = [];      // SampleWithObsAct objects

        // Training session state - EXACT from Java: inTrainSess = false
        // User must toggle training ON with spacebar before giving rewards
        this.inTrainSess = false;

        // Debug mode - set to true to log credit assignment details
        this.debug = false;
    }

    /**
     * Enable/disable debug logging
     */
    setDebug(enabled) {
        this.debug = enabled;
        console.log(`CreditAssign debug mode: ${enabled ? 'ON' : 'OFF'}`);
    }

    /**
     * TimeStepForCred equivalent - stores timestep data
     */
    _createTimeStep() {
        return {
            startTime: Number.NEGATIVE_INFINITY,
            endTime: Number.POSITIVE_INFINITY,
            feats: null,
            obs: null,
            action: null,
            throwOut: false,
            setInStone: false,
            credUsedBeforeLastStep: 0
        };
    }

    /**
     * SampleWithObsAct equivalent - stores sample data
     */
    _createSample(feats, weight, unique, obs, action) {
        return {
            feats: feats ? [...feats] : null,
            label: 0,
            weight: weight,
            unique: unique,
            obs: obs ? [...obs] : null,
            action: action,
            unweightedRew: 0,
            usedCredit: 0,
            creditUsedLastStep: 0
        };
    }

    /**
     * Record the end of a timestep - EXACT from Java recordTimeStepEnd()
     * @param {number} btwnStepTime - Time in seconds
     */
    recordTimeStepEnd(btwnStepTime) {
        const lastTimeStepI = this.timeStepsInWindow.length - 1;
        if (this.timeStepsInWindow.length > 0) {
            this.timeStepsInWindow[lastTimeStepI].endTime = btwnStepTime;
        }
    }

    /**
     * Record the start of a timestep - EXACT from Java recordTimeStepStart()
     * @param {number[]} feats - Feature vector
     * @param {number} btwnStepTime - Time in seconds
     * @param {number[]} obs - Observation (optional)
     * @param {number} action - Action (optional)
     */
    recordTimeStepStart(feats, btwnStepTime, obs = null, action = null) {
        // Add new time step
        const newStep = this._createTimeStep();
        this.timeStepsInWindow.push(newStep);

        const lastTimeStepI = this.timeStepsInWindow.length - 1;

        // Verify previous step ended
        if (this.timeStepsInWindow.length > 1 &&
            this.timeStepsInWindow[lastTimeStepI - 1].endTime === Number.POSITIVE_INFINITY) {
            console.error('Tried to create a new time step before ending the last.');
        }

        this.timeStepsInWindow[lastTimeStepI].feats = feats ? [...feats] : null;
        this.timeStepsInWindow[lastTimeStepI].startTime = btwnStepTime;
        this.timeStepsInWindow[lastTimeStepI].obs = obs;
        this.timeStepsInWindow[lastTimeStepI].action = action;

        // Create new sample
        const newSampleUnique = this.delayWtedIndivRew ? -1 : this.totalTimeSteps + this.UNIQUE_START;
        this.activeSamples.push(this._createSample(feats, 1.0, newSampleUnique, obs, action));

        this.totalTimeSteps++;
    }

    /**
     * Get steps before current - EXACT from Java getStepsBeforeCurrent()
     */
    _getStepsBeforeCurrent(stepIndex) {
        return (this.timeStepsInWindow.length - stepIndex) - 1;
    }

    /**
     * Process a human reward signal - EXACT from Java processNewHReward()
     * @param {number} hReward - The reward value
     * @param {number} hRewTime - Time when reward was given (in seconds)
     */
    processNewHReward(hReward, hRewTime) {
        // Debug: log credit assignment
        if (this.debug) {
            console.log(`\n--- processNewHReward: reward=${hReward} at t=${hRewTime.toFixed(3)}s ---`);
            console.log(`Credit window: [${(hRewTime - this.windowEnd).toFixed(3)}s, ${(hRewTime - this.windowStart).toFixed(3)}s]`);
            console.log(`Active timesteps: ${this.timeStepsInWindow.length}`);
        }

        for (let i = 0; i < this.timeStepsInWindow.length; i++) {
            const credit = this._getCredit(hRewTime, i, this._getStepsBeforeCurrent(i));
            const rewardShare = hReward * credit;

            const sample = this.activeSamples[i];
            sample.unweightedRew += rewardShare;
            sample.usedCredit = Math.max(
                this._getCreditPastElig(i, this._getStepsBeforeCurrent(i), hRewTime),
                sample.usedCredit
            );
            sample.label = sample.unweightedRew;

            if (this.EXTRAPOLATE_FUTURE_REW) {
                sample.label /= sample.usedCredit;
            }

            // Debug: log credit for each step
            if (this.debug && credit > 0) {
                const step = this.timeStepsInWindow[i];
                console.log(`  Step ${i}: ${step.startTime.toFixed(3)}s-${step.endTime === Infinity ? 'ongoing' : step.endTime.toFixed(3) + 's'} -> credit=${credit.toFixed(4)}, label=${sample.label.toFixed(4)}`);
            }
        }
    }

    /**
     * Get relative near bound - EXACT from Java getRelNearBound()
     */
    _getRelNearBound(stepI, currTime) {
        if (this.timeStepsInWindow[stepI].endTime !== Number.NEGATIVE_INFINITY) {
            return currTime - this.timeStepsInWindow[stepI].endTime;
        }
        return 0.0;
    }

    /**
     * Calculate credit for a timestep - EXACT from Java getCredit()
     */
    _getCredit(hRewTime, stepI, stepsBeforeCurrent) {
        let credit = 0;

        if (this.distClass === 'previousStep') {
            if (stepsBeforeCurrent === 1) {
                credit = 1;
            } else {
                credit = 0;
            }
            return credit;
        } else if (this.distClass === 'immediate') {
            if (stepsBeforeCurrent === 0) {
                credit = 1;
            } else {
                credit = 0;
            }
            return credit;
        } else if (this.distClass === 'uniform') {
            let relativeFarBound = hRewTime - this.timeStepsInWindow[stepI].startTime;
            let relativeNearBound = this._getRelNearBound(stepI, hRewTime);

            if ((relativeFarBound > this.windowStart) && (relativeNearBound < this.windowEnd)) {
                if (relativeNearBound < this.windowStart) {
                    relativeNearBound = this.windowStart;
                }
                if (relativeFarBound > this.windowEnd) {
                    relativeFarBound = this.windowEnd;
                }
                credit = (relativeFarBound - relativeNearBound) / (this.windowEnd - this.windowStart);
            } else {
                credit = 0;
            }
        } else {
            console.error('Using an invalid distribution class for credit assignment!');
        }

        return credit;
    }

    /**
     * Calculate credit past eligibility - EXACT from Java getCreditPastElig()
     * Complex parallelogram calculation for uniform distribution
     */
    _getCreditPastElig(sampleI, stepsBeforeCurrent, currTime, duringStepTransition = false) {
        if (this.distClass === 'previousStep') {
            return this._isSampleFinished(sampleI, duringStepTransition) ? 1.0 : 0.0;
        } else if (this.distClass === 'immediate') {
            return this._isSampleFinished(sampleI, duringStepTransition) ? 1.0 : 0.0;
        } else if (this.distClass === 'uniform') {
            const step = this.timeStepsInWindow[sampleI];
            const uniformWdthOfSupport = this.windowEnd - this.windowStart;

            // When step isn't finished yet, assume it ends now
            const stepWidth = Math.min(step.endTime, currTime) - step.startTime;

            // If stepWidth is 0, the steps are shorter than can be represented
            if (stepWidth === 0) {
                if (step.startTime >= currTime) {
                    return 0;
                }
                if (this.inTrainSess) {
                    console.error('Steps of zero duration during training.');
                }
                return 1.0;
            }

            // Time since crediting started
            const t = currTime - (step.startTime + this.windowStart);

            // Parallelogram calculations - EXACT from Java
            const pgramRectLength = Math.abs(uniformWdthOfSupport - stepWidth);
            const pgramTriSide = Math.min(uniformWdthOfSupport, stepWidth);
            const pgramArea = pgramTriSide * (pgramTriSide + pgramRectLength);

            // Calculate areas
            const rampUpArea = Math.pow(Math.min(pgramTriSide, Math.max(0, t)), 2) / 2.0;
            const rectArea = Math.max(0, Math.min(pgramRectLength, t - pgramTriSide)) * pgramTriSide;
            const rampDnArea = Math.max(0,
                Math.pow(Math.min(pgramTriSide, Math.max(0, t - (pgramTriSide + pgramRectLength))), 2) / 2.0
            );

            const pastArea = rampUpArea + rectArea + rampDnArea;
            const creditPastElig = (pgramArea === 0.0) ? 0.0 : Math.min(1.0, pastArea / pgramArea);

            return creditPastElig;
        } else {
            console.error('Using an invalid distribution class for credit assignment!');
            return 0.0;
        }
    }

    /**
     * Check if sample is finished - EXACT from Java isSampleFinished()
     */
    _isSampleFinished(sampleI, duringStepTransition = false) {
        if (this.timeStepsInWindow[sampleI].setInStone) {
            return true;
        }

        if (this.distClass === 'previousStep' || this.distClass === 'immediate') {
            const stepsBeforeCurrent = this._getStepsBeforeCurrent(sampleI);
            let finishedBound = 0;

            if (this.distClass === 'immediate') {
                finishedBound = duringStepTransition ? 0 : 1;
            } else if (this.distClass === 'previousStep') {
                finishedBound = duringStepTransition ? 1 : 2;
            }

            if (stepsBeforeCurrent === finishedBound) {
                return true;
            }
        } else if (this.distClass === 'uniform') {
            const usedCredit = this.activeSamples[sampleI].usedCredit;
            if (usedCredit <= 1 && usedCredit >= this.APPROX_ONE) {
                return true;
            }
        }

        return false;
    }

    /**
     * Process timesteps - EXACT from Java processTimeSteps()
     */
    _processTimeSteps(currTime, inTrainSess, duringStepTransition = false) {
        const activeCreditedSamples = [];

        for (let i = 0; i < this.timeStepsInWindow.length; i++) {
            const activeSample = this.activeSamples[i];
            const step = this.timeStepsInWindow[i];

            if (step.setInStone) {
                continue;
            }

            const priorUsedCredit = activeSample.usedCredit;

            // Update sample for this time step
            activeSample.usedCredit = this._getCreditPastElig(i, this._getStepsBeforeCurrent(i), currTime, duringStepTransition);
            activeSample.label = activeSample.unweightedRew;

            if (this.EXTRAPOLATE_FUTURE_REW) {
                activeSample.label /= activeSample.usedCredit;
            }

            activeSample.creditUsedLastStep = activeSample.usedCredit - step.credUsedBeforeLastStep;

            // If first credit opportunity is missed, throw out sample
            if (!inTrainSess && activeSample.usedCredit > 0.0 &&
                (priorUsedCredit === 0.0 || priorUsedCredit === Number.NEGATIVE_INFINITY)) {
                step.throwOut = true;
            }

            if (!inTrainSess && this._isSampleFinished(i, duringStepTransition) &&
                (this.distClass === 'previousStep' || this.distClass === 'immediate')) {
                step.throwOut = true;
            }

            // Add unfinished samples for possible model update
            if (this.distClass !== 'immediate' && this.distClass !== 'previousStep' &&
                activeSample.usedCredit > this.MIN_USED_CRED_FOR_EXTRAP &&
                activeSample.usedCredit < this.APPROX_ONE &&
                !step.throwOut) {
                activeCreditedSamples.push(activeSample);
            }

            step.credUsedBeforeLastStep = activeSample.usedCredit;
        }

        return activeCreditedSamples;
    }

    /**
     * Remove finished timesteps - EXACT from Java removeFinishedTimeSteps()
     */
    _removeFinishedTimeSteps(currTime, inTrainSess, duringStepTransition = false) {
        const removedSamples = [];
        const iRemovalOffset = this.distClass === 'immediate' ? 0 : 1;

        for (let i = 0; i < this.timeStepsInWindow.length - iRemovalOffset; i++) {
            if (this.timeStepsInWindow[i].throwOut) {
                this._removeSample(i);
                i--;
            } else if (this._isSampleFinished(i, duringStepTransition)) {
                removedSamples.push(this._removeSample(i));
                i--;
            }
        }

        return removedSamples;
    }

    /**
     * Remove a sample - EXACT from Java removeSample()
     */
    _removeSample(sampleI) {
        this.timeStepsInWindow.splice(sampleI, 1);
        const sample = this.activeSamples.splice(sampleI, 1)[0];
        sample.label = sample.unweightedRew / sample.usedCredit;
        return sample;
    }

    /**
     * Process samples and remove finished - EXACT from Java processSamplesAndRemoveFinished()
     * @param {number} currTime - Current time in seconds
     * @param {boolean} inTrainSess - Whether in training session
     * @param {boolean} duringStepTransition - True if called during agent_step (affects previousStep/immediate finish bounds)
     * @returns {Array} Samples ready for model update
     */
    processSamplesAndRemoveFinished(currTime, inTrainSess, duringStepTransition = false) {
        const activeCreditedSamples = this._processTimeSteps(currTime, inTrainSess, duringStepTransition);
        const removedSamples = this._removeFinishedTimeSteps(currTime, inTrainSess, duringStepTransition);

        if (this.noUpdateWhenNoRew) {
            if (this.delayWtedIndivRew && this._allSamplesHaveZeroRew()) {
                return [];
            }
        }

        let samples = [];

        if (this.EXTRAPOLATE_FUTURE_REW || this.delayWtedIndivRew) {
            samples = samples.concat(activeCreditedSamples);

            if (this.delayWtedIndivRew) {
                // Clone samples and reset
                for (let i = 0; i < this.activeSamples.length; i++) {
                    const cloned = { ...this.activeSamples[i] };
                    cloned.feats = [...this.activeSamples[i].feats];
                    cloned.unweightedRew = 0;
                    cloned.label = 0;
                    this.activeSamples[i] = cloned;
                }
            }
        }

        samples = samples.concat(removedSamples);

        if (this.delayWtedIndivRew) {
            this._removeSamplesWNoNewCred(samples);
            this._setWtToCredLastStep(samples);
        } else {
            if (this.noUpdateWhenNoRew) {
                this._removeSamplesWZeroRew(samples);
            }
        }

        return samples;
    }

    /**
     * Helper methods - EXACT from Java
     */
    _removeSamplesWNoNewCred(samples) {
        for (let i = samples.length - 1; i >= 0; i--) {
            if (samples[i].creditUsedLastStep === 0.0) {
                samples.splice(i, 1);
            }
        }
    }

    _removeSamplesWZeroRew(samples) {
        for (let i = samples.length - 1; i >= 0; i--) {
            if (samples[i].label === 0) {
                samples.splice(i, 1);
            }
        }
    }

    _setWtToCredLastStep(samples) {
        for (const sample of samples) {
            sample.weight = sample.creditUsedLastStep;
        }
    }

    _allSamplesHaveZeroRew() {
        for (const sample of this.activeSamples) {
            if (sample.unweightedRew !== 0.0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Set training session state - EXACT from Java setInTrainSess()
     */
    setInTrainSess(currTime, newInTrainSess) {
        if (this.inTrainSess && !newInTrainSess) {
            for (let i = 0; i < this.timeStepsInWindow.length; i++) {
                const usedUpCredit = this._getCreditPastElig(i, this._getStepsBeforeCurrent(i), currTime);

                if (usedUpCredit < this.SAMPLE_CUMUL_CRED_MIN && usedUpCredit > 0.0) {
                    this.timeStepsInWindow[i].throwOut = true;
                } else if (usedUpCredit >= this.SAMPLE_CUMUL_CRED_MIN) {
                    this.timeStepsInWindow[i].setInStone = true;
                    this.activeSamples[i].usedCredit = usedUpCredit;
                }
            }
        }

        this.inTrainSess = newInTrainSess;
    }

    /**
     * Clear history - EXACT from Java clearHistory()
     */
    clearHistory() {
        this.timeStepsInWindow = [];
        this.activeSamples = [];
    }

    /**
     * Toggle training session
     */
    toggleTraining() {
        const currTime = performance.now() / 1000;
        this.setInTrainSess(currTime, !this.inTrainSess);
        return this.inTrainSess;
    }

    /**
     * Check if in training session
     */
    isTraining() {
        return this.inTrainSess;
    }

    /**
     * Draw a random delay - EXACT from Java drawDelay()
     */
    drawDelay() {
        if (this.distClass === 'previousStep') {
            return 0;
        } else if (this.distClass === 'uniform') {
            return this.windowStart + (Math.random() * (this.windowEnd - this.windowStart));
        }
        return 0;
    }
}
